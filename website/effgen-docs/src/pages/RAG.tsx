import { Library } from 'lucide-react';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function RAG() {
  return (
    <DocPage
      subtitle="Indexing documents, retrieving passages, and answering with citations back to them."
      icon={<Library size={48} />}
    >
      <p>
        Point an agent at a directory and it answers from what is in it, with a citation on every
        claim. Underneath that is a pipeline you can take apart — ingestion, chunking, hybrid
        search, reranking, context building and attribution — and use one stage of on its own.
      </p>

      <h2>The whole thing, as one preset</h2>

      <CodeBlock filename="rag.py" code={`import pathlib

from effgen import create_agent

docs = pathlib.Path("/tmp/effgen-kb")
docs.mkdir(exist_ok=True)
(docs / "architecture.md").write_text(
    "# Scaling\\n\\n"
    "effGen scales horizontally: run one server per replica behind a load balancer.\\n"
    "State lives in the session store, so any replica can answer any request.\\n"
)
(docs / "storage.md").write_text(
    "# Storage\\n\\nSessions are JSON files under ~/.effgen/sessions.\\n"
)

agent = create_agent("rag", "gpt-5-nano", provider="openai", knowledge_base=str(docs))

response = agent.run("What does the architecture document say about scaling?")
print(response.text)
print("sources:", response.sources)`} />

      <Terminal
        command="python rag.py"
        output={`The architecture document describes horizontal scaling for effGen: run one server per replica behind a load balancer, with state stored in a session store so any replica can handle requests. [1]
sources: ['/tmp/effgen-kb/architecture.md']`}
        caption={`Run against effGen 1.0.0. The [1] marker is the agent's own citation, and response.sources names the file it came from.`}
      />

      <ApiTable
        headers={['On the response', 'What it holds']}
        rows={[
          [
            <code>response.text</code>,
            <>
              The answer, with inline [1], [2] markers where a passage was used. The{' '}
              <code>rag</code> preset asks for them; any other agent writes them only with{' '}
              <code>cite_sources=True</code>, on <code>AgentConfig</code> or on{' '}
              <code>run()</code>. When it does, <code>[n]</code> is{' '}
              <code>citations[n - 1]</code>.
            </>,
          ],
          [
            <code>response.citations</code>,
            <>
              One <code>Citation</code> per marker — <code>index</code>, <code>source</code>,{' '}
              <code>chunk_id</code>, <code>relevance_score</code>, <code>quote</code>,{' '}
              <code>page</code>, <code>section</code>.
            </>,
          ],
          [<code>response.sources</code>, 'The source files, deduplicated.'],
        ]}
      />

      <h2>Ingesting documents</h2>

      <CodeBlock filename="ingest.py" code={`from effgen.rag import DocumentIngester

# dedupe is on by default; it drops chunks with an identical content hash.
chunks = DocumentIngester(chunk_size=500, chunk_overlap=100).ingest(
    "/tmp/effgen-kb", recursive=True
)

print(len(chunks), "chunks")
for chunk in chunks:
    print(" ", chunk.source, "|", chunk.content[:60].replace("\\n", " "))`} />

      <Terminal command="python ingest.py" output={`2 chunks
  /tmp/effgen-kb/storage.md | # Storage  Sessions are JSON files under ~/.effgen/sessions.
  /tmp/effgen-kb/architecture.md | # Scaling  effGen scales horizontally: run one server per re`} />

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'chunker',
            type: 'Any',
            default: 'None',
            description: 'A chunker instance. Without one, fixed-size chunking is used.',
          },
          { name: 'chunk_size', type: 'int', default: '500', description: 'Target characters per chunk.' },
          { name: 'chunk_overlap', type: 'int', default: '100', description: 'How much of the previous chunk each one repeats.' },
          {
            name: 'dedupe',
            type: 'bool',
            default: 'True',
            description: 'Drop chunks with an identical content hash. On by default.',
          },
          { name: 'show_progress', type: 'bool', default: 'True', description: 'Draw a progress bar while reading. Turn it off in a script.' },
        ]}
        caption={
          <>
            <code>DocumentIngester(...)</code>; then{' '}
            <code>ingest(source, recursive=True)</code>, which takes a path or a list of paths.
          </>
        }
      />

      <ApiTable
        headers={['Formats', 'Which']}
        rows={[
          ['Built in', <><code>txt</code>, <code>md</code>, <code>json</code>, <code>jsonl</code>, <code>csv</code>, <code>html</code></>],
          [
            'With an extra package',
            <>
              <code>pdf</code> (pypdf), <code>docx</code> (python-docx), <code>epub</code>{' '}
              (ebooklib)
            </>,
          ],
        ]}
        caption={
          <>
            Each chunk carries <code>id</code>, <code>content</code>, <code>source</code>,{' '}
            <code>metadata</code> and <code>content_hash</code>.
          </>
        }
      />

      <h3>Chunking that knows what it is reading</h3>

      <CodeBlock filename="chunking.py" code={`from effgen.rag.chunking import CodeChunker, HierarchicalChunker, TextChunker

SOURCE = (open("/tmp/effgen-kb/architecture.md").read() + "\\n") * 6

for chunker in (
    TextChunker(chunk_size=200, overlap=20),
    HierarchicalChunker(max_chunk_size=200),
    CodeChunker(language="python", max_chunk_size=200),
):
    chunks = chunker.chunk(SOURCE)
    print(f"{type(chunker).__name__:20} {len(chunks):2} chunks, "
          f"longest {max(len(c.content) for c in chunks)} chars")`} />

      <Terminal
        command="python chunking.py"
        output={`TextChunker           6 chunks, longest 173 chars
HierarchicalChunker   6 chunks, longest 160 chars
CodeChunker           1 chunks, longest 982 chars`}
        caption="The same text through three chunkers. CodeChunker looks for Python structure, finds none in a Markdown file, and falls back to one chunk — which is the right answer for the wrong input."
      />

      <ApiTable
        headers={['Chunker', 'Splits on']}
        rows={[
          [<code>TextChunker</code>, 'A fixed size, with overlap. The default.'],
          [<code>SemanticChunker</code>, 'Semantic boundaries rather than a character count.'],
          [
            <code>CodeChunker</code>,
            <>
              Functions, classes and blocks. <code>language=</code> takes py, js, ts, go, rust or
              java.
            </>,
          ],
          [<code>TableChunker</code>, 'Table boundaries, so a table is not cut in half.'],
          [<code>HierarchicalChunker</code>, 'Document structure, keeping the heading a chunk sits under.'],
        ]}
      />

      <h2>Hybrid search</h2>
      <p>
        Dense retrieval, BM25, keyword matching and metadata filtering, fused with reciprocal rank
        fusion. Every result says what each of those contributed.
      </p>

      <CodeBlock filename="search.py" code={`from effgen.rag import DocumentIngester, HybridSearchEngine

chunks = DocumentIngester(show_progress=False).ingest("/tmp/effgen-kb", recursive=True)
engine = HybridSearchEngine(chunks)

for result in engine.search("how does it scale", top_k=2):
    print(result.rank, round(result.relevance_score, 3), "|", result.source)
    print("   dense", round(result.dense_score, 3),
          "sparse", round(result.sparse_score, 3),
          "keyword", round(result.keyword_score, 3))`} />

      <Terminal command="python search.py" output={`1 1.0 | /tmp/effgen-kb/architecture.md
   dense 1.0 sparse 0.0 keyword 0.0
2 0.984 | /tmp/effgen-kb/storage.md
   dense 0.021 sparse 0.0 keyword 0.0`} />

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'chunk_id', type: 'str', description: 'Which chunk this is.' },
          { name: 'content', type: 'str', description: 'The passage itself.' },
          { name: 'source', type: 'str', description: 'The file it came from.' },
          { name: 'metadata', type: 'dict', description: 'Whatever the ingester recorded about it.' },
          { name: 'relevance_score', type: 'float', description: 'The fused score — what the ranking is on.' },
          { name: 'dense_score', type: 'float', description: 'The embedding similarity alone.' },
          { name: 'sparse_score', type: 'float', description: 'The BM25 score alone.' },
          { name: 'keyword_score', type: 'float', description: 'The literal keyword match alone.' },
          { name: 'rank', type: 'int', description: 'Its position in this result set.' },
        ]}
        caption={<><code>effgen.rag.SearchResult</code>. A result that scored on one signal and not the others is visible rather than hidden behind one number.</>}
      />

      <h3>Weighting the signals</h3>

      <CodeBlock filename="weights.py" code={`from effgen.rag import DocumentIngester, HybridSearchEngine

chunks = DocumentIngester(show_progress=False).ingest("/tmp/effgen-kb", recursive=True)

# The weights belong to the engine, not to a single search.
engine = HybridSearchEngine(
    chunks,
    weights={"dense": 0.4, "bm25": 0.3, "keyword": 0.2, "metadata": 0.1},
)
for result in engine.search("how does it scale", top_k=2):
    print(round(result.relevance_score, 3), "|", result.source)`} />

      <Terminal command="python weights.py" output={`1.0 | /tmp/effgen-kb/architecture.md
0.984 | /tmp/effgen-kb/storage.md`} />

      <Callout type="note" title="The weights belong to the engine">
        <p>
          They are a constructor argument on <code>HybridSearchEngine</code>, not an argument to{' '}
          <code>search()</code>. <code>search(query, top_k=5, filter_metadata=None, min_score=0.0)</code>{' '}
          is the whole of the search signature.
        </p>
      </Callout>

      <h2>Reranking</h2>

      <CodeBlock filename="rerank.py" code={`from effgen.rag import DocumentIngester, HybridSearchEngine
from effgen.rag.reranker import RuleBasedReranker

chunks = DocumentIngester(show_progress=False).ingest("/tmp/effgen-kb", recursive=True)
results = HybridSearchEngine(chunks).search("how does it scale", top_k=4)

reranker = RuleBasedReranker(keyword_weight=0.4, authority_weight=0.1, recency_weight=0.0)
for result in reranker.rerank("how does it scale", results):
    print(round(result.relevance_score, 3), "|", result.source)`} />

      <Terminal command="python rerank.py" output={`1.133 | /tmp/effgen-kb/architecture.md
0.984 | /tmp/effgen-kb/storage.md`} />

      <ApiTable
        headers={['Reranker', 'How it scores', 'Cost']}
        rows={[
          [
            <code>RuleBasedReranker</code>,
            <>
              Weighted signals: <code>recency_weight</code>, <code>keyword_weight</code>,{' '}
              <code>authority_weight</code>, <code>title_weight</code>, plus an{' '}
              <code>authority_map</code> per source.
            </>,
            'Free, no model',
          ],
          [
            <code>LLMReranker</code>,
            <>
              Asks a model to order the passages. Takes the model, and{' '}
              <code>max_passage_chars</code>.
            </>,
            'One model call',
          ],
          [
            <code>CrossEncoderReranker</code>,
            <>
              A cross-encoder, <code>cross-encoder/ms-marco-MiniLM-L-6-v2</code> by default.
            </>,
            <>Local inference; needs <code>sentence-transformers</code></>,
          ],
        ]}
        caption={
          <>
            All three share one signature:{' '}
            <code>rerank(query, results, top_k=None)</code> — the query first.
          </>
        }
      />

      <h2>Building the context, and the citations</h2>

      <CodeBlock filename="context.py" code={`from effgen.rag import ContextBuilder, DocumentIngester, HybridSearchEngine

chunks = DocumentIngester(show_progress=False).ingest("/tmp/effgen-kb", recursive=True)
results = HybridSearchEngine(chunks).search("how does it scale", top_k=2)

built = ContextBuilder(max_tokens=2048).build(results)
print(built.text[:200])
print("---", built.total_tokens, "tokens, truncated:", built.truncated)
for citation in built.citations:
    print(f"[{citation.index}]", citation.source, "|", round(citation.relevance_score, 3))`} />

      <Terminal command="python context.py" output={`[1] Source: /tmp/effgen-kb/architecture.md
# Scaling

effGen scales horizontally: run one server per replica behind a load balancer.
State lives in the session store, so any replica can answer any req
--- 77 tokens, truncated: False
[1] /tmp/effgen-kb/architecture.md | 1.0
[2] /tmp/effgen-kb/storage.md | 0.984`} />

      <ParamTable
        nameLabel="Argument"
        params={[
          { name: 'max_tokens', type: 'int', default: '2000', description: 'The budget the assembled context has to fit in.' },
          {
            name: 'token_counter',
            type: 'Callable | None',
            default: 'None',
            description: 'How tokens are counted. Falls back to an estimate.',
          },
          {
            name: 'per_source_limit',
            type: 'int',
            default: '1',
            description: 'How many chunks one source may contribute, so one document cannot crowd out the rest.',
          },
          {
            name: 'order',
            type: 'str',
            default: "'relevance'",
            description: 'The order passages are laid out in.',
          },
          { name: 'include_citations', type: 'bool', default: 'True', description: 'Whether [n] markers are written into the text.' },
          { name: 'separator', type: 'str', default: "'\\n\\n---\\n\\n'", description: 'What goes between passages.' },
        ]}
        caption={
          <>
            <code>build(results)</code> returns a <code>BuiltContext</code> — <code>text</code>,{' '}
            <code>citations</code>, <code>used_chunks</code>, <code>total_tokens</code>,{' '}
            <code>truncated</code>. It is one object, not a tuple.
          </>
        }
      />

      <h3>Checking the citations against the answer</h3>

      <CodeBlock
        filename="attribution.py"
        continues
        code={`from effgen.rag.attribution import CitationTracker

tracker = CitationTracker(built.citations)

answer = "effGen scales horizontally, one server per replica behind a load balancer [1]."

used = tracker.extract_used_indices(answer)
print("cited:", used)
for citation in tracker.filter_used(answer):
    print(f"  [{citation.index}]", citation.source)

supported = tracker.verify("effGen scales horizontally behind a load balancer")
print("passages supporting that claim:", [c.index for c in supported])
print("sources:", tracker.sources())`}
        caption="Carries on from context.py above, which builds the citations. Verification is what catches a model citing a passage it did not use."
      />

      <Terminal
        command="python attribution.py"
        output={`cited: [1]
  [1] /tmp/effgen-kb/architecture.md
passages supporting that claim: [1]
sources: ['/tmp/effgen-kb/architecture.md', '/tmp/effgen-kb/storage.md']`}
      />

      <p>
        <code>verify()</code> takes one claim and returns the citations that support it, not a
        boolean — a claim nothing backs comes back as an empty list. The threshold it compares
        against is <code>min_overlap</code>, which defaults to <code>0.3</code>.
      </p>

      <h2>Retrieval as a tool</h2>
      <p>
        The same index, exposed to an agent as an ordinary tool it can decide to call — rather than
        as context that is always prepended.
      </p>

      <CodeBlock filename="retrieval_tool.py" code={`import asyncio

from effgen.tools.builtin.retrieval import Retrieval

tool = Retrieval(knowledge_base_path="/tmp/effgen-kb", default_top_k=2)
result = asyncio.run(tool.execute(query="how does effGen scale"))

if not result.success:
    raise SystemExit(result.error)
for hit in result.output["results"]:
    print(round(hit["score"], 3), "|", hit["content"][:70].replace("\\n", " "))`} />

      <Terminal command="python retrieval_tool.py" output={`0.885 | # Scaling  effGen scales horizontally: run one server per replica behi
0.646 | # Storage  Sessions are JSON files under ~/.effgen/sessions. `} />

      <ParamTable
        nameLabel="Argument"
        params={[
          { name: 'embedding_provider', type: 'EmbeddingProvider | None', default: 'None', description: 'What turns text into vectors.' },
          { name: 'chunk_size', type: 'int', default: '500', description: 'Target characters per chunk.' },
          { name: 'chunk_overlap', type: 'int', default: '100', description: 'Overlap between chunks.' },
          { name: 'chunking_strategy', type: 'str', default: "'fixed'", description: 'How documents are split.' },
          { name: 'index_path', type: 'str | None', default: 'None', description: 'Where a built index is cached.' },
          { name: 'knowledge_base_path', type: 'str | None', default: 'None', description: 'The documents to index.' },
          { name: 'enable_hybrid_search', type: 'bool', default: 'True', description: 'Combine dense and sparse retrieval rather than dense alone.' },
          { name: 'hybrid_alpha', type: 'float', default: '0.7', description: 'How the two are weighted against each other.' },
          {
            name: 'allow_pickle',
            type: 'bool',
            default: 'False',
            description: 'Whether a pickled index may be loaded. Off, because a pickle from an untrusted source executes code.',
          },
          { name: 'default_top_k', type: 'int', default: '5', description: 'How many passages a call returns unless it asks for more.' },
          { name: 'diversity', type: 'float', default: '0.0', description: 'How much to penalise near-duplicate passages in one result set.' },
        ]}
        caption={
          <>
            <code>effgen.tools.builtin.retrieval.Retrieval</code>. The companion{' '}
            <code>agentic_search</code> tool does the same job with exact string matching, which
            beats semantic search for a number, a formula or an identifier.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'Zero chunks from ingestion',
            'The path is wrong, or nothing in it is in a supported format.',
            <>
              Check the path, and pass <code>recursive=True</code> for subdirectories. PDF, DOCX
              and EPUB each need their own package.
            </>,
          ],
          [
            <><code>TypeError: cannot unpack non-iterable BuiltContext object</code></>,
            <>
              <code>build()</code> was unpacked as a tuple.
            </>,
            <>
              It returns one object: <code>built.text</code>, <code>built.citations</code>.
            </>,
          ],
          [
            <><code>TypeError: search() got an unexpected keyword argument 'weights'</code></>,
            'The weights were passed to the search rather than to the engine.',
            <>
              <code>HybridSearchEngine(chunks, weights=&#123;…&#125;)</code>.
            </>,
          ],
          [
            'The reranker returns nothing useful',
            <>
              Its arguments are the other way round: <code>rerank(query, results)</code>.
            </>,
            'The query comes first on all three rerankers.',
          ],
          [
            'The answer has no citations',
            <>
              <code>include_citations</code> is off, or the model was given plain context.
            </>,
            <>
              The <code>rag</code> preset wires the markers and the citation list. Check{' '}
              <code>response.citations</code> before concluding it did not cite.
            </>,
          ],
          [
            'Retrieval finds nothing for an exact term',
            'Semantic search is the wrong instrument for an identifier, a number or a formula.',
            <>
              Use the <code>agentic_search</code> tool, which matches literally, or raise{' '}
              <code>keyword</code> in the engine weights.
            </>,
          ],
          [
            'One document dominates every answer',
            <>
              <code>per_source_limit</code> is doing less than you want.
            </>,
            <>
              Raise <code>diversity</code> on the retrieval tool, or lower{' '}
              <code>per_source_limit</code> on the context builder.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/memory', '/tools/gallery', '/presets']} />
    </DocPage>
  );
}
