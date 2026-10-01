import { Library } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function Domains() {
  return (
    <DocPage
      subtitle="Domain packs: the prompt, tools and settings a subject area is set up with."
      icon={<Library size={48} />}
    >
      <p>
        A <code>Domain</code> is a small bundle of subject-matter setup — a system prompt, the tools
        that subject actually needs, a guardrail choice, and a set of seed keywords with the query
        shapes that subject is usually asked in. Five ship. Turning one into a working agent is one
        call.
      </p>

      <h2>One call to an agent</h2>

      <CodeBlock filename="legal.py" code={`from effgen.domains import LegalDomain

agent = LegalDomain().to_agent(model="gpt-5-nano", provider="openai")
print("agent name :", agent.config.name)
print("tools      :", [t.name for t in agent.config.tools])
print("guardrails :", agent.config.guardrails)
print("prompt     :", agent.config.system_prompt[:60], "…")`} />

      <Terminal
        command="python legal.py"
        output={`agent name : legal-agent
tools      : ['web_search', 'wikipedia']
guardrails : standard
prompt     : You are a legal information expert. Explain legal concepts,  …`}
        caption={`Run against effGen 1.0.0. The two tools, the guardrail preset and the prompt all came from the domain — nothing else was configured.`}
      />

      <h2>What is in one</h2>

      <CodeBlock filename="tech.py" code={`from effgen.domains import TechDomain

domain = TechDomain()
print("name       :", domain.name)
print("description:", domain.description)
print("tools      :", domain.tool_names)
print("keywords   :", len(domain.keywords), "seeds, first five:", domain.keywords[:5])
print("prompt     :", domain.system_prompt[:72], "…")`} />

      <Terminal
        command="python tech.py"
        output={`name       : tech
description: Software engineering, programming languages, DevOps, and cloud computing.
tools      : ['code_executor', 'python_repl', 'web_search', 'bash']
keywords   : 20 seeds, first five: ['Python', 'JavaScript', 'TypeScript', 'Rust', 'Go']
prompt     : You are a technology expert specializing in software engineering, progra …`}
      />

      <h2>The five that ship</h2>

      <CodeBlock filename="all.py" code={`from effgen.domains import (
    FinanceDomain, HealthDomain, LegalDomain, ScienceDomain, TechDomain,
)

for cls in (TechDomain, FinanceDomain, HealthDomain, LegalDomain, ScienceDomain):
    d = cls()
    print(f"{d.name:8} {len(d.keywords):2} keywords  guardrails={d.guardrails!s:9} tools={','.join(d.tool_names)}")`} />

      <Terminal
        command="python all.py"
        output={`tech     20 keywords  guardrails=None      tools=code_executor,python_repl,web_search,bash
finance  20 keywords  guardrails=None      tools=calculator,web_search,python_repl
health   20 keywords  guardrails=standard  tools=web_search,wikipedia,calculator
legal    20 keywords  guardrails=standard  tools=web_search,wikipedia
science  20 keywords  guardrails=None      tools=calculator,python_repl,web_search,wikipedia`}
        caption={
          <>
            Health and legal carry the <code>standard</code> guardrail preset and no execution tools;
            tech carries <code>bash</code> and <code>code_executor</code> and no guardrails. The
            difference is the point — see <Link to="/guardrails">Guardrails</Link>.
          </>
        }
      />

      <ApiTable
        headers={['Domain', 'Covers', 'Set up with']}
        rows={[
          [
            <code>TechDomain</code>,
            'Software engineering, programming languages, DevOps, cloud.',
            <>
              Code execution and a shell. Its prompt asks for up-to-date technical answers with code
              where it helps.
            </>,
          ],
          [
            <code>FinanceDomain</code>,
            'Markets, banking, cryptocurrency, personal finance.',
            'A calculator, search and a Python REPL. Its prompt asks it to note that it is not financial advice.',
          ],
          [
            <code>HealthDomain</code>,
            'Medical science, wellness, nutrition, public health.',
            <>
              Search, Wikipedia, a calculator, and the <code>standard</code> guardrails. Its prompt
              asks for evidence-based answers and requires the not-medical-advice note.
            </>,
          ],
          [
            <code>LegalDomain</code>,
            'Law, regulations, compliance.',
            <>
              Search and Wikipedia, and the <code>standard</code> guardrails. No execution tools at
              all.
            </>,
          ],
          [
            <code>ScienceDomain</code>,
            'Physics, chemistry, biology, astronomy.',
            'A calculator, a Python REPL, search and Wikipedia. Its prompt asks for established theory and formulas.',
          ],
        ]}
        caption={
          <>
            All five from <code>effgen.domains</code>, each a subclass of <code>Domain</code> with
            its fields filled in. Each carries twenty seed keywords and nine query templates.
          </>
        }
      />

      <Callout type="note" title="A domain is not a preset">
        <p>
          A <Link to="/presets">preset</Link> answers "what kind of work is this" — maths, coding,
          RAG, multimodal — and there are nine of them, reachable from the command line. A domain
          answers "what subject is this about", and it is a Python object you can build and edit. A
          domain's <code>to_agent()</code> produces an ordinary agent, so the two compose: pass a
          preset's tools, or override anything the domain set.
        </p>
      </Callout>

      <h2>Growing a keyword list</h2>
      <p>
        <code>expand_keywords</code> takes the seeds and the domain's query templates and produces the
        phrasings that subject is actually searched in — useful for seeding a{' '}
        <Link to="/rag">retrieval corpus</Link>, an evaluation set, or a crawl.
      </p>

      <CodeBlock filename="expand.py" code={`from effgen.domains import ScienceDomain

domain = ScienceDomain(keywords=["photosynthesis", "enzyme kinetics"])
expanded = domain.expand_keywords(factor=6)
print(len(expanded), "queries from 2 seeds")
for q in expanded[:8]:
    print("  ", q)`} />

      <Terminal
        command="python expand.py"
        output={`20 queries from 2 seeds
   enzyme kinetics
   enzyme kinetics applications
   enzyme kinetics equation
   enzyme kinetics examples
   enzyme kinetics experiment
   enzyme kinetics mechanism
   enzyme kinetics principles
   enzyme kinetics research`}
        caption={
          <>
            <code>factor</code> is a target, not a promise — the templates and the deduplication
            decide the real count. Science's templates are theory, experiment, mechanism, research,
            applications, principles, equation, examples and a comparison.
          </>
        }
      />

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'factor',
            type: 'int',
            default: '10',
            description: 'How many queries to aim for per seed keyword.',
          },
          {
            name: 'use_templates',
            type: 'bool',
            default: 'True',
            description: 'Apply the domain’s query templates. Keyword-only.',
          },
          {
            name: 'use_wordnet',
            type: 'bool',
            default: 'False',
            description: 'Add WordNet synonyms. Opt-in, and needs the corpus installed. Keyword-only.',
          },
          {
            name: 'use_llm',
            type: 'bool',
            default: 'False',
            description: 'Ask a model for further phrasings. Costs a call. Keyword-only.',
          },
          {
            name: 'model',
            type: 'Any',
            default: 'None',
            description: 'The model to use when use_llm is on. Keyword-only.',
          },
        ]}
        caption={
          <>
            <code>Domain.expand_keywords</code>. <code>KeywordExpander</code> takes the same
            switches, plus <code>templates</code>, and exposes <code>expand(keywords, factor=10)</code>{' '}
            for use without a domain.
          </>
        }
      />

      <h2>Writing your own</h2>
      <p>
        <code>Domain</code> is a dataclass. Building one is filling in the fields, and it works the
        same as the five that ship.
      </p>

      <CodeBlock filename="support.py" code={`from effgen.domains import Domain

support = Domain(
    name="support",
    description="Tier-one product support for the billing system.",
    keywords=["refund", "invoice", "chargeback"],
    system_prompt="You answer billing questions. Quote the policy you relied on.",
    tool_names=["web_search", "calculator"],
    guardrails="phi",
)
print(support.to_dict())`} />

      <Terminal
        command="python support.py"
        output={`{'name': 'support', 'keywords': ['refund', 'invoice', 'chargeback'], 'description': 'Tier-one product support for the billing system.', 'system_prompt': 'You answer billing questions. Quote the policy you relied on.', 'tool_names': ['web_search', 'calculator']}`}
        caption={
          <>
            <code>to_dict()</code> carries the five fields that describe the domain. The guardrail
            choice is applied when the agent is built, not serialised here.
          </>
        }
      />

      <ParamTable
        nameLabel="Field"
        params={[
          {
            name: 'name',
            type: 'str',
            required: true,
            description: 'Short id. The agent built from it is named "<name>-agent".',
          },
          {
            name: 'keywords',
            type: 'list[str]',
            default: '[]',
            description: 'Seed terms. What expand_keywords starts from.',
          },
          {
            name: 'description',
            type: 'str',
            default: "''",
            description: 'One sentence saying what the subject covers.',
          },
          {
            name: 'system_prompt',
            type: 'str',
            default: '"You are a helpful AI assistant."',
            description: 'The prompt the agent is built with.',
          },
          {
            name: 'tool_names',
            type: 'list[str]',
            default: '[]',
            description: 'Built-in tools by name, resolved when the agent is built.',
          },
          {
            name: 'guardrails',
            type: 'Any',
            default: 'None',
            description: (
              <>
                A preset name or a <code>GuardrailChain</code>, passed straight to{' '}
                <code>AgentConfig</code>.
              </>
            ),
          },
          {
            name: 'templates',
            type: 'list[str]',
            default: '[]',
            description: (
              <>
                Query shapes for keyword expansion, with <code>{'{kw}'}</code> and{' '}
                <code>{'{alt}'}</code> placeholders.
              </>
            ),
          },
          { name: 'metadata', type: 'dict[str, Any]', default: '{}', description: 'Free space.' },
        ]}
        caption={<><code>Domain</code>, from <code>effgen.domains</code>.</>}
      />

      <ApiTable
        headers={['Method', 'Returns']}
        rows={[
          [
            <code>to_agent(model=None, **overrides)</code>,
            <>
              An <code>Agent</code> built from the domain. Any keyword argument overrides what the
              domain set.
            </>,
          ],
          [
            <code>expand_keywords(factor=10, *, use_wordnet, use_templates, use_llm, model)</code>,
            <code>list[str]</code>,
          ],
          [<code>to_dict()</code>, 'The domain as plain data.'],
        ]}
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>RuntimeError: Failed to load model '…'</code>,
            <>
              <code>to_agent</code> builds a real agent, so it needs a usable model and a key.
            </>,
            <>
              Pass <code>model=</code> and <code>provider=</code>, and check the key is loaded. See{' '}
              <Link to="/models">Models</Link>.
            </>,
          ],
          [
            <><code>agent.config.guardrails</code> is a string, not a chain</>,
            'A domain stores the preset name. The agent resolves it when it is constructed.',
            <>
              That is the shape. <code>AgentConfig(guardrails="standard")</code> is a supported way
              to configure guardrails.
            </>,
          ],
          [
            'Fewer expanded keywords than factor × seeds',
            'The templates ran out, or duplicates were removed.',
            <>
              Add templates of your own, or turn on <code>use_wordnet</code> /{' '}
              <code>use_llm</code>.
            </>,
          ],
          [
            <><code>use_wordnet=True</code> adds nothing</>,
            'The WordNet corpus is not installed. It is an opt-in extra, and its absence is not an error.',
            'Install it, or leave the switch off.',
          ],
          [
            'A domain agent has no tools',
            <>
              A name in <code>tool_names</code> did not resolve to a built-in tool.
            </>,
            <>
              <code>effgen tools list</code> has the names. See{' '}
              <Link to="/tools/gallery">the tool gallery</Link>.
            </>,
          ],
          [
            'A health or legal agent refuses an ordinary question',
            <>
              Both carry the <code>standard</code> guardrail chain, which screens input as well as
              output.
            </>,
            <>
              Override it: <code>to_agent(model=…, guardrails="minimal")</code>. Consider why it
              matched first.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/presets', '/guardrails', '/agents']} />
    </DocPage>
  );
}
