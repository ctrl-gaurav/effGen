"use client";

import { useState, useEffect, useRef, useCallback, type ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FiMessageCircle, FiX, FiSend, FiMail, FiGithub, FiChevronRight, FiArrowLeft } from "react-icons/fi";
import { usePyPIVersion } from "./PyPIVersion";
import { highlightCode } from "./syntaxHighlight";
import { faqs, type FAQ } from "./helpBotFaqs";
import { withBasePath } from "./basePath";

// The answer corpus lives in helpBotFaqs.ts — it is the part that goes stale at
// a release, and keeping it apart from the matching engine below means a version
// bump is one file to walk rather than fifteen hundred lines to read past.
const getFAQById = (id: number, fallbackIndex = 0): FAQ =>
  faqs.find((faq) => faq.id === id) ?? faqs[fallbackIndex];

// ---------------------------------------------------------------------------
// Matching engine – TF-IDF-style scoring with bigrams + exact-phrase bonus
// ---------------------------------------------------------------------------

// Simple stemming: strip common English suffixes so "installing" matches "install"
const suffixes = ["ing", "tion", "ment", "ness", "able", "ible", "ous", "ive", "ful", "less", "ly", "ed", "er", "es", "s"];
function stem(word: string): string {
  for (const s of suffixes) {
    if (word.length > s.length + 2 && word.endsWith(s)) {
      return word.slice(0, -s.length);
    }
  }
  return word;
}

// English stop-words to ignore
const STOP = new Set([
  "the","a","an","is","it","in","on","to","of","and","or","for","with",
  "that","this","what","how","do","does","i","my","me","can","you","your",
  "be","are","was","were","have","has","had","will","would","could","should",
  "if","but","not","no","yes","so","as","at","by","from","up","out","about",
  "which","who","when","where","why","they","them","their","we","our",
  "can","may","might","must","shall","am","been","being","get","got",
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 1 && !STOP.has(w))
    .map(stem);
}

// Build bigrams from a token array
function bigrams(tokens: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < tokens.length - 1; i++) {
    out.push(tokens[i] + " " + tokens[i + 1]);
  }
  return out;
}

// Pre-compute IDF weights across the FAQ corpus
function buildIDF(docs: string[][]): Map<string, number> {
  const df = new Map<string, number>();
  const N = docs.length;
  for (const tokens of docs) {
    const unique = new Set(tokens);
    unique.forEach((t) => df.set(t, (df.get(t) ?? 0) + 1));
  }
  const idf = new Map<string, number>();
  df.forEach((count, term) => {
    idf.set(term, Math.log(N / count));
  });
  return idf;
}

// Tokenised FAQ corpus (unigrams + bigrams) – built once at module load
const faqTokenised = faqs.map((faq) => {
  const base = tokenize(faq.question + " " + faq.tags.join(" ") + " " + faq.answer);
  return [...base, ...bigrams(base)];
});
const IDF = buildIDF(faqTokenised);

// Cosine-style TF-IDF score between query and a single FAQ
function scoreFAQ(queryTokens: string[], faqTokens: string[]): number {
  const tfFaq = new Map<string, number>();
  faqTokens.forEach((t) => tfFaq.set(t, (tfFaq.get(t) ?? 0) + 1));

  let dot = 0, magQ = 0, magF = 0;
  const allTerms = new Set([...queryTokens, ...faqTokens]);
  allTerms.forEach((t) => {
    const idf = IDF.get(t) ?? 1;
    const qTF = queryTokens.filter((x) => x === t).length;
    const fTF = tfFaq.get(t) ?? 0;
    const qW = qTF * idf;
    const fW = fTF * idf;
    dot += qW * fW;
    magQ += qW * qW;
    magF += fW * fW;
  });

  return magQ === 0 || magF === 0 ? 0 : dot / (Math.sqrt(magQ) * Math.sqrt(magF));
}

// Exact-phrase bonus: if any multi-word tag appears verbatim in the query, boost hard
function exactPhraseBonus(queryLower: string, faq: FAQ): number {
  let bonus = 0;
  for (const tag of faq.tags) {
    if (tag.includes(" ") && queryLower.includes(tag)) {
      bonus += 0.25; // significant boost per exact multi-word match
    }
  }
  return bonus;
}

interface ScoredFAQ {
  faq: FAQ;
  score: number;
}

function findBestMatches(query: string, topK = 3): ScoredFAQ[] {
  const queryLower = query.toLowerCase();
  const queryTokens = tokenize(query);
  const queryBi = bigrams(queryTokens);
  const allQueryTokens = [...queryTokens, ...queryBi];

  const scored: ScoredFAQ[] = faqs.map((faq, i) => ({
    faq,
    score: scoreFAQ(allQueryTokens, faqTokenised[i]) + exactPhraseBonus(queryLower, faq),
  }));

  scored.sort((a, b) => b.score - a.score);

  return scored.filter((s) => s.score > 0.08).slice(0, topK);
}

// Two kinds of link appear in an answer, and they must not be treated the same.
// A site path such as /docs/quickstart stays inside the site and follows the
// base path; anything else is somewhere else and opens in its own tab. The
// previous corpus wrote its documentation links as absolute
// https://www.effgen.org/docs/... URLs, and every one of them 404'd.
const SITE_PATHS = /^\/(docs|changelog|examples|community|leaderboard|cli|code|dashboard|models|agents|production)(\/|$)/;

function renderLinks(text: string, keyPrefix: string): ReactNode[] {
  const urlPattern =
    /(https?:\/\/[^\s<>()]+|(?:github\.com|discord\.gg|effgen\.org|arxiv\.org)\/[^\s<>()]+|\/(?:docs|changelog|examples|community|leaderboard|cli|code|dashboard|models|agents|production)(?:\/[a-z0-9-]+)*)/g;

  return text.split(urlPattern).filter(Boolean).map((part, index) => {
    if (SITE_PATHS.test(part)) {
      return (
        <a
          key={`${keyPrefix}-link-${index}`}
          href={withBasePath(part)}
          className="text-green-700 dark:text-green-400 underline underline-offset-2"
        >
          {part}
        </a>
      );
    }

    const isUrl = /^(https?:\/\/|github\.com\/|discord\.gg\/|effgen\.org\/|arxiv\.org\/)/.test(part);
    if (!isUrl) return part;

    const href = part.startsWith("http") ? part : `https://${part}`;
    return (
      <a
        key={`${keyPrefix}-link-${index}`}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-green-700 dark:text-green-400 underline underline-offset-2"
      >
        {part}
      </a>
    );
  });
}

function renderInlineMarkdown(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  text.split("**").forEach((part, index) => {
    if (index % 2 === 1) {
      nodes.push(
      <strong key={`${keyPrefix}-strong-${index}`}>
        {renderLinks(part, `${keyPrefix}-strong-${index}`)}
      </strong>
      );
    } else {
      nodes.push(...renderLinks(part, `${keyPrefix}-text-${index}`));
    }
  });
  return nodes;
}

function renderMarkdownText(text: string, keyPrefix: string): ReactNode[] {
  return text
    .split(/\n{2,}/)
    .filter((block) => block.trim().length > 0)
    .map((block, blockIndex) => {
      const lines = block.split("\n").filter((line) => line.trim().length > 0);
      const allBullets = lines.every((line) => /^\s*[-*]\s+/.test(line));
      const allOrdered = lines.every((line) => /^\s*\d+\.\s+/.test(line));

      if (allBullets) {
        return (
          <ul key={`${keyPrefix}-ul-${blockIndex}`} className="my-2 ml-5 list-disc space-y-1">
            {lines.map((line, lineIndex) => (
              <li key={`${keyPrefix}-li-${blockIndex}-${lineIndex}`}>
                {renderInlineMarkdown(line.replace(/^\s*[-*]\s+/, ""), `${keyPrefix}-li-${blockIndex}-${lineIndex}`)}
              </li>
            ))}
          </ul>
        );
      }

      if (allOrdered) {
        return (
          <ol key={`${keyPrefix}-ol-${blockIndex}`} className="my-2 ml-5 list-decimal space-y-1">
            {lines.map((line, lineIndex) => (
              <li key={`${keyPrefix}-oli-${blockIndex}-${lineIndex}`}>
                {renderInlineMarkdown(line.replace(/^\s*\d+\.\s+/, ""), `${keyPrefix}-oli-${blockIndex}-${lineIndex}`)}
              </li>
            ))}
          </ol>
        );
      }

      return (
        <p key={`${keyPrefix}-p-${blockIndex}`} className="my-1.5 whitespace-pre-line leading-relaxed">
          {renderInlineMarkdown(block, `${keyPrefix}-p-${blockIndex}`)}
        </p>
      );
    });
}

// ---------------------------------------------------------------------------
// Category grouping helper for the browse view
// ---------------------------------------------------------------------------
const CATEGORIES = [...new Set(faqs.map((f) => f.category))];

// ---------------------------------------------------------------------------
// Chat UI state
// ---------------------------------------------------------------------------
type MessageType = "user" | "bot" | "suggestions" | "contact-prompt" | "contact-options";

interface Message {
  type: MessageType;
  content: string;
  suggestions?: ScoredFAQ[];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function HelpBot() {
  const { version: pypiVersion } = usePyPIVersion();
  const [isOpen, setIsOpen] = useState(false);
  const [view, setView] = useState<"chat" | "browse">("chat"); // chat vs category browse
  const [browseCategory, setBrowseCategory] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      type: "bot",
      content: "Ask a question about effGen 1.3.0, or browse the topics below. Everything here links into the documentation.",
    },
    {
      type: "suggestions",
      content: "",
      // The six openers. Chosen for what people arrive not knowing: how to
      // install it, how to write the first agent, what changed in 1.3.0, and
      // the two 1.0.0 surfaces that are hardest to guess at — pointing it at
      // your own server, and the coding agent.
      suggestions: [
        { faq: getFAQById(1), score: 1 },    // How do I install effGen?
        { faq: getFAQById(10), score: 1 },   // How do I create my first agent?
        { faq: getFAQById(64), score: 1 },   // What's new in 1.3.0?
        { faq: getFAQById(6), score: 1 },    // Point it at my own vLLM/Ollama server
        { faq: getFAQById(24), score: 1 },   // What is effgen code?
        { faq: getFAQById(17), score: 1 },   // What built-in tools are there?
      ],
    },
  ]);
  const [input, setInput] = useState("");
  const [isWaitingForContact, setIsWaitingForContact] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 100);
  }, [isOpen]);

  // ── core send handler ─────────────────────────────────────────────────────
  const handleSend = useCallback(() => {
    if (!input.trim()) return;
    const userQuery = input.trim();
    setInput("");

    // If we just asked "would you like to contact support?" and the user says yes/no
    if (isWaitingForContact) {
      setIsWaitingForContact(false);
      setMessages((prev) => [...prev, { type: "user", content: userQuery }]);
      const yes = /^(yes|y|yeah|yep|sure|ok|okay|please)\b/i.test(userQuery);
      if (yes) {
        setTimeout(() => {
          setMessages((prev) => [
            ...prev,
            { type: "contact-options", content: "" },
          ]);
        }, 300);
      } else {
        setTimeout(() => {
          setMessages((prev) => [
            ...prev,
            { type: "bot", content: "No problem! Feel free to ask another question anytime." },
          ]);
        }, 300);
      }
      return;
    }

    // Normal flow
    setMessages((prev) => [...prev, { type: "user", content: userQuery }]);

    const matches = findBestMatches(userQuery, 3);

    setTimeout(() => {
      if (matches.length > 0 && matches[0].score > 0.12) {
        // Good match – show the best answer
        setMessages((prev) => [
          ...prev,
          { type: "bot", content: matches[0].faq.answer },
        ]);
        // Show related if we have them
        if (matches.length > 1) {
          setTimeout(() => {
            setMessages((prev) => [
              ...prev,
              { type: "bot", content: "**Related questions:**" },
              {
                type: "suggestions",
                content: "",
                suggestions: matches.slice(1),
              },
            ]);
          }, 400);
        }
      } else {
        // No good match – offer contact, but DON'T loop
        setIsWaitingForContact(true);
        setMessages((prev) => [
          ...prev,
          {
            type: "bot",
            content: "I don't have a great answer for that one. You can:\n\n1. **Rephrase** your question and try again\n2. **Browse topics** using the menu above\n3. **Contact us** directly (I can show you how)",
          },
          { type: "contact-prompt", content: "Would you like me to show you the contact options?" },
        ]);
      }
    }, 300);
  }, [input, isWaitingForContact]);

  const handleSuggestionClick = (faq: FAQ) => {
    setIsWaitingForContact(false);
    setMessages((prev) => [
      ...prev,
      { type: "user", content: faq.question },
      { type: "bot", content: faq.answer },
    ]);
  };

  // ── render helpers ────────────────────────────────────────────────────────
  const renderBotContent = (rawContent: string) => {
    const content = rawContent.replace(/__VERSION__/g, pypiVersion);
    return content.split("```").flatMap((part, i) => {
      if (i % 2 === 1) {
        const lines = part.split("\n");
        const hasLang = !!lines[0].match(/^[a-z]+$/);
        const lang = hasLang ? lines[0] : "python";
        const code = hasLang ? lines.slice(1).join("\n") : part;
        return (
          <pre key={i} className="bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-100 p-2.5 rounded-lg my-2 text-xs overflow-x-auto font-mono">
            <code
              className="syntax-code"
              dangerouslySetInnerHTML={{ __html: highlightCode(code, lang) }}
            />
          </pre>
        );
      }
      return renderMarkdownText(part, `part-${i}`);
    });
  };

  // ── browse view ───────────────────────────────────────────────────────────
  const renderBrowse = () => {
    if (browseCategory) {
      const items = faqs.filter((f) => f.category === browseCategory);
      return (
        <div className="flex flex-col h-full">
          <button
            onClick={() => setBrowseCategory(null)}
            className="flex items-center gap-1.5 text-xs text-green-700 dark:text-green-400 hover:underline px-4 pt-3 pb-1"
          >
            <FiArrowLeft size={12} /> Back to topics
          </button>
          <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1.5">
            {items.map((faq) => (
              <button
                key={faq.id}
                onClick={() => {
                  setView("chat");
                  handleSuggestionClick(faq);
                }}
                className="w-full text-left px-3 py-2.5 bg-gray-50 dark:bg-gray-800/50 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-xl text-sm text-gray-700 dark:text-gray-300 transition-colors flex items-start gap-2 group"
              >
                <FiChevronRight className="text-green-500 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" size={13} />
                <span className="group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors">{faq.question}</span>
              </button>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="flex flex-col h-full">
        <p className="text-xs text-gray-600 dark:text-gray-400 px-4 pt-3 pb-1">Choose a topic</p>
        <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1.5">
          {CATEGORIES.map((cat) => {
            const count = faqs.filter((f) => f.category === cat).length;
            return (
              <button
                key={cat}
                onClick={() => setBrowseCategory(cat)}
                className="w-full text-left px-3 py-2.5 bg-gray-50 dark:bg-gray-800/50 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-xl text-sm text-gray-700 dark:text-gray-300 transition-colors flex items-center justify-between group"
              >
                <span className="group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors font-medium">{cat}</span>
                <span className="text-xs text-gray-400 dark:text-gray-400 bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded-full">{count}</span>
              </button>
            );
          })}
        </div>
      </div>
    );
  };

  // ── main render ───────────────────────────────────────────────────────────
  return (
    <>
      {/* Trigger button */}
      <motion.button
        onClick={() => setIsOpen(true)}
        aria-label="Open the help panel"
        className="fixed bottom-24 right-8 w-14 h-14 rounded-full bg-gradient-to-r from-green-600 to-emerald-600 flex items-center justify-center text-white shadow-lg hover:shadow-xl hover:shadow-emerald-500/30 transition-all z-40"
        whileHover={{ scale: 1.1, y: -2 }}
        whileTap={{ scale: 0.95 }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
      >
        <FiMessageCircle size={24} aria-hidden="true" />
      </motion.button>

      {/* Chat window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-24 right-8 w-96 h-[540px] bg-white dark:bg-gray-900 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 flex flex-col overflow-hidden z-50"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white flex-shrink-0">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">
                  <FiMessageCircle size={16} />
                </div>
                <div>
                  <h3 className="font-semibold text-sm">effGen Help</h3>
                  <p className="text-xs text-white/70">{faqs.length} topics · Ask anything</p>
                </div>
              </div>
              <div className="flex items-center gap-1">
                {/* Chat / Browse toggle */}
                <div className="flex bg-white/15 rounded-lg p-0.5 mr-1">
                  <button
                    onClick={() => setView("chat")}
                    className={`px-2 py-0.5 rounded-md text-xs font-medium transition-colors ${view === "chat" ? "bg-white text-green-700" : "text-white/80 hover:text-white"}`}
                  >
                    Chat
                  </button>
                  <button
                    onClick={() => { setView("browse"); setBrowseCategory(null); }}
                    className={`px-2 py-0.5 rounded-md text-xs font-medium transition-colors ${view === "browse" ? "bg-white text-green-700" : "text-white/80 hover:text-white"}`}
                  >
                    Browse
                  </button>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1.5 rounded-lg hover:bg-white/20 transition-colors"
                >
                  <FiX size={18} />
                </button>
              </div>
            </div>

            {/* Body – swaps between chat and browse */}
            {view === "browse" ? (
              <div className="flex-1 overflow-hidden">{renderBrowse()}</div>
            ) : (
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.map((message, index) => (
                  <div key={index}>
                    {message.type === "user" && (
                      <div className="flex justify-end">
                        <div className="max-w-[80%] bg-gradient-to-r from-green-600 to-emerald-600 text-white px-4 py-2 rounded-2xl rounded-br-md text-sm">
                          {message.content}
                        </div>
                      </div>
                    )}

                    {message.type === "bot" && (
                      <div className="flex justify-start">
                        <div className="max-w-[85%] bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 px-4 py-2.5 rounded-2xl rounded-bl-md text-sm whitespace-pre-wrap">
                          {renderBotContent(message.content)}
                        </div>
                      </div>
                    )}

                    {message.type === "suggestions" && message.suggestions && (
                      <div className="space-y-1.5 mt-1">
                        {message.suggestions.map((scored, i) => (
                          <motion.button
                            key={i}
                            onClick={() => handleSuggestionClick(scored.faq)}
                            className="w-full text-left px-3 py-2 bg-gray-50 dark:bg-gray-800/50 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-xl text-sm text-gray-700 dark:text-gray-300 transition-colors flex items-center gap-2 group border border-gray-200 dark:border-gray-700/50 hover:border-green-300 dark:hover:border-green-700/50"
                            whileHover={{ x: 3 }}
                          >
                            <FiChevronRight className="text-green-500 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" size={14} />
                            <span className="group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors">
                              {scored.faq.question}
                            </span>
                          </motion.button>
                        ))}
                      </div>
                    )}

                    {message.type === "contact-prompt" && (
                      <div className="flex justify-start mt-1">
                        <div className="max-w-[85%] bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-700/40 text-green-800 dark:text-green-200 px-4 py-2.5 rounded-2xl rounded-bl-md text-sm">
                          {message.content}
                        </div>
                      </div>
                    )}

                    {message.type === "contact-options" && (
                      <div className="flex justify-start mt-1">
                        <div className="max-w-[90%] bg-gray-100 dark:bg-gray-800 px-4 py-3 rounded-2xl rounded-bl-md text-sm space-y-2">
                          <p className="text-gray-700 dark:text-gray-300 font-medium">Here&apos;s how to reach us:</p>
                          <a
                            href="mailto:gks@vt.edu"
                            className="flex items-center gap-2 px-3 py-2 bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-green-400 dark:hover:border-green-500 transition-colors group"
                          >
                            <FiMail size={14} className="text-green-700 dark:text-green-400" />
                            <div className="text-left">
                              <p className="text-xs font-semibold text-gray-800 dark:text-gray-200">Email</p>
                              <p className="text-xs text-gray-600 dark:text-gray-400 group-hover:text-green-500 dark:group-hover:text-green-400 transition-colors">gks@vt.edu</p>
                            </div>
                          </a>
                          <a
                            href="https://github.com/ctrl-gaurav/effGen/issues/new"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-3 py-2 bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-green-400 dark:hover:border-green-500 transition-colors group"
                          >
                            <FiGithub size={14} className="text-green-700 dark:text-green-400" />
                            <div className="text-left">
                              <p className="text-xs font-semibold text-gray-800 dark:text-gray-200">GitHub Issue</p>
                              <p className="text-xs text-gray-600 dark:text-gray-400 group-hover:text-green-500 dark:group-hover:text-green-400 transition-colors">Report a bug or request a feature</p>
                            </div>
                          </a>
                          <p className="text-xs text-gray-400 dark:text-gray-400 pt-1">We typically respond within 24–48 hours.</p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}

            {/* Bottom contact strip – always visible */}
            <div className="px-3 py-2 border-t border-gray-200 dark:border-gray-700 flex gap-2 flex-shrink-0">
              <a
                href="mailto:gks@vt.edu"
                className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 rounded-lg text-xs text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <FiMail size={12} /> Email
              </a>
              <a
                href="https://github.com/ctrl-gaurav/effGen/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 rounded-lg text-xs text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <FiGithub size={12} /> Issues
              </a>
            </div>

            {/* Input row – hidden in browse mode */}
            {view === "chat" && (
              <div className="px-3 py-2.5 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
                <div className="flex items-center gap-2">
                  <input
                    ref={inputRef}
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    placeholder="Type your question..."
                    className="flex-1 px-4 py-2 bg-gray-100 dark:bg-gray-800 rounded-xl text-sm text-gray-800 dark:text-gray-200 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:bg-white dark:focus:bg-gray-700 transition-colors"
                  />
                  <motion.button
                    onClick={handleSend}
                    disabled={!input.trim()}
                    className="p-2 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl disabled:opacity-40 disabled:cursor-not-allowed shadow-sm hover:shadow-md transition-shadow"
                    whileHover={input.trim() ? { scale: 1.08 } : {}}
                    whileTap={input.trim() ? { scale: 0.95 } : {}}
                  >
                    <FiSend size={16} />
                  </motion.button>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
