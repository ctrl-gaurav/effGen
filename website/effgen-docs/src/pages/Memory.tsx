import { Brain } from 'lucide-react';
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

export default function Memory() {
  return (
    <DocPage
      subtitle="Short-term and long-term memory, what each stores, and when either is consulted."
      icon={<Brain size={48} />}
    >
      <p>
        An agent's memory is three separate stores with three separate jobs: the conversation it is
        having now, facts it should keep after the conversation ends, and a vector index for
        recalling something by meaning rather than by keyword. Only the first is on by default.
      </p>

      <h2>The conversation an agent is having</h2>

      <CodeBlock filename="remembers.py" code={`from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    name="assistant",
    model="gpt-5-nano",
    provider="openai",
    enable_memory=True,
))

agent.run("My dog is named Pixel.")
print(agent.run("What is my dog's name?").text)`} />

      <Terminal
        command="python remembers.py"
        output={`Pixel`}
        caption={`Run against effGen 1.0.0.`}
      />

      <p>
        <code>enable_memory=True</code> gives the agent a <code>ShortTermMemory</code>, and every
        turn goes into it. It lives in the process — when the process ends, so does it. To carry a
        conversation across runs or across machines, use a{' '}
        <Link to="/sessions">session</Link> instead.
      </p>

      <h2>Short-term memory</h2>

      <CodeBlock filename="short_term.py" code={`from effgen.memory import ShortTermMemory

memory = ShortTermMemory(max_tokens=4096, max_messages=100, keep_recent_messages=10)
memory.add_user_message("My dog is named Pixel.")
memory.add_assistant_message("Noted — Pixel it is.")
memory.add_tool_message("weather → 18.0 Clear sky")

print(memory.get_token_count(), "tokens over", len(memory.get_messages()), "messages")
for message in memory.get_messages():
    print(f"  {message.role.value:9} {message.content[:40]!r}")
print(memory.get_statistics())`} />

      <Terminal command="python short_term.py" output={`16 tokens over 3 messages
  user      'My dog is named Pixel.'
  assistant 'Noted — Pixel it is.'
  tool      'weather → 18.0 Clear sky'
{'current_messages': 3, 'current_tokens': 16, 'max_tokens': 4096, 'utilization': 0.00390625, 'total_messages_added': 3, 'total_summarizations': 0, 'summaries_count': 0}`} />

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'max_tokens',
            type: 'int',
            default: '4096',
            description: 'The window the history has to fit in. Past a fraction of it, compaction runs.',
          },
          { name: 'max_messages', type: 'int', default: '100', description: 'A hard cap on how many messages are held.' },
          {
            name: 'summarization_threshold',
            type: 'float',
            default: '0.8',
            description: 'The fraction of max_tokens at which compaction is triggered.',
          },
          {
            name: 'summary_length_ratio',
            type: 'float',
            default: '0.3',
            description: 'How long a generated summary may be, relative to what it replaces.',
          },
          {
            name: 'keep_recent_messages',
            type: 'int',
            default: '10',
            description: 'How many of the most recent turns are never compacted.',
          },
          {
            name: 'summary_budget_ratio',
            type: 'float',
            default: '0.4',
            description: 'The share of the window summaries are allowed to occupy.',
          },
          {
            name: 'model',
            type: 'Any',
            default: 'None',
            description: 'The model that writes summaries. Without one, a strategy that needs a model cannot summarise.',
          },
          {
            name: 'compaction_strategy',
            type: 'Any',
            default: 'None',
            description: (
              <>
                Which turns leave and what replaces them — see{' '}
                <Link to="/compaction">Context compaction</Link>. Defaults to{' '}
                <code>SummarizeOldest</code>.
              </>
            ),
          },
          {
            name: 'tokenizer',
            type: 'Any',
            default: 'None',
            description: 'Anything with count_tokens(text) or encode(text). Without one the history is estimated at four characters per token.',
          },
        ]}
        caption={<><code>effgen.memory.ShortTermMemory</code></>}
      />

      <ApiTable
        headers={['Method', 'What it does']}
        rows={[
          [
            <>
              <code>add_user_message(content)</code>, <code>add_assistant_message(content)</code>,{' '}
              <code>add_system_message(content)</code>, <code>add_tool_message(content)</code>
            </>,
            <>
              Append one turn. Each returns the <code>Message</code> it stored.
            </>,
          ],
          [
            <code>add_message(role, content, metadata=None, tokens=None)</code>,
            <>
              The general form. <code>role</code> is a <code>MessageRole</code>.
            </>,
          ],
          [<code>get_messages()</code>, 'Everything currently held, in order.'],
          [<code>get_recent_messages(n)</code>, 'The last n.'],
          [<code>get_messages_by_role(role)</code>, <>Filtered by <code>MessageRole</code>.</>],
          [<code>search_messages(text)</code>, 'Every message whose content contains that text.'],
          [<code>get_conversation_context()</code>, 'The history rendered as prompt text.'],
          [<code>get_token_count()</code>, 'How large the history currently is.'],
          [<code>get_statistics()</code>, 'Messages held, tokens, utilisation, and how many times it has compacted.'],
          [<code>clear()</code>, 'Forget everything.'],
          [
            <>
              <code>save_to_file(path)</code>, <code>ShortTermMemory.load_from_file(path)</code>
            </>,
            <>
              Round-trip to JSON. <code>load_from_file</code> is a class method and returns a new
              memory.
            </>,
          ],
          [<>
            <code>to_dict()</code>, <code>from_dict(data)</code>
          </>, 'The same, without touching the filesystem.'],
        ]}
      />

      <h3>Finding something in it</h3>

      <CodeBlock filename="search.py" code={`from effgen.memory import MessageRole, ShortTermMemory

memory = ShortTermMemory()
memory.add_user_message("My dog is named Pixel.")
memory.add_user_message("My cat is named Mote.")
memory.add_assistant_message("Two pets noted.")

for message in memory.search_messages("dog"):
    print(message.role.value, "|", message.content)
print([m.content for m in memory.get_messages_by_role(MessageRole.USER)])`} />

      <Terminal command="python search.py" output={`user | My dog is named Pixel.
['My dog is named Pixel.', 'My cat is named Mote.']`} />

      <h3>Saving and restoring it</h3>

      <CodeBlock filename="persist.py" code={`from effgen.memory import ShortTermMemory

memory = ShortTermMemory()
memory.add_user_message("My dog is named Pixel.")
memory.save_to_file("/tmp/effgen-short-term.json")

restored = ShortTermMemory.load_from_file("/tmp/effgen-short-term.json")
print([m.content for m in restored.get_messages()])`} />

      <Terminal command="python persist.py" output={`['My dog is named Pixel.']`} />

      <Callout type="note" title="load_from_file is a class method">
        <p>
          It returns a new <code>ShortTermMemory</code> rather than filling one in place, so{' '}
          <code>ShortTermMemory.load_from_file(path)</code> is the call. Assigning it to an
          existing memory does nothing to that memory.
        </p>
      </Callout>

      <h2>Long-term memory</h2>
      <p>
        A store for facts that outlive one conversation, with a type and an importance on every
        entry, kept in SQLite or JSON. It is not consulted automatically — you write to it and read
        from it.
      </p>

      <CodeBlock filename="long_term.py" code={`from effgen.memory import (
    ImportanceLevel, LongTermMemory, MemoryType, SQLiteStorageBackend,
)

memory = LongTermMemory(backend=SQLiteStorageBackend("/tmp/effgen-memories.db"))
memory.start_session("user-123")

memory.add_memory(
    content="Pixel is the user's dog, a border collie.",
    memory_type=MemoryType.FACT,
    importance=ImportanceLevel.HIGH,
    tags=["pets"],
)

for entry in memory.search("dog", limit=3):
    print(entry.memory_type.value, entry.importance.name, "|", entry.content)

print(memory.get_statistics())
memory.end_session()
memory.close()`} />

      <Terminal command="python long_term.py" output={`fact HIGH | Pixel is the user's dog, a border collie.
{'total_memories': 1, 'max_memories': 10000, 'current_session_id': 'c8252769-69b3-440d-aa5a-8fc0dc03b3d3', 'total_consolidations': 0, 'memories_by_type': {'fact': 1}, 'memories_by_importance': {'HIGH': 1}}`} />

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'backend',
            type: 'StorageBackend',
            required: true,
            description: 'SQLiteStorageBackend(path) or JSONStorageBackend(path). There is no default.',
          },
          {
            name: 'consolidation_interval',
            type: 'int',
            default: '100',
            description: 'How many additions between consolidation passes.',
          },
          { name: 'max_memories', type: 'int', default: '10000', description: 'The cap. Past it, the least important go first.' },
          {
            name: 'min_importance_to_keep',
            type: 'ImportanceLevel',
            default: 'ImportanceLevel.LOW',
            description: 'The floor consolidation keeps. Anything below it can be dropped.',
          },
        ]}
        caption={<><code>effgen.memory.LongTermMemory</code></>}
      />

      <ApiTable
        headers={['Call', 'What it does']}
        rows={[
          [
            <code>add_memory(content, memory_type, importance=…, tags=None, metadata=None)</code>,
            <>
              Store one fact and return the <code>MemoryEntry</code>. Note that these are keyword
              arguments, not a constructed entry.
            </>,
          ],
          [
            <code>search(query=None, memory_type=None, session_id=None, tags=None, min_importance=None, limit=50)</code>,
            <>
              Every filter is optional and they combine. Returns a list of{' '}
              <code>MemoryEntry</code>.
            </>,
          ],
          [<code>get_memory(entry_id)</code>, 'One entry by id, bumping its access count.'],
          [
            <>
              <code>start_session(id)</code>, <code>end_session()</code>
            </>,
            'Tag everything added between the two with a session, so it can be searched back out.',
          ],
          [<code>consolidate()</code>, 'Merge duplicates and drop what is below the importance floor.'],
          [<code>get_statistics()</code>, 'Totals, and the breakdown by type and by importance.'],
          [<code>clear_all()</code>, 'Empty the store.'],
          [<code>close()</code>, 'Close the backend. Do this, or a SQLite file can be left locked.'],
        ]}
      />

      <ApiTable
        headers={['MemoryType', 'For']}
        rows={[
          [<code>conversation</code>, 'A turn worth keeping past the conversation.'],
          [<code>fact</code>, 'Something true about the user or the domain.'],
          [<code>observation</code>, 'Something the agent noticed.'],
          [<code>task</code>, 'Work in progress, or work that was done.'],
          [<code>tool_result</code>, 'Output worth not fetching twice.'],
          [<code>reflection</code>, 'The agent’s own note about how a run went.'],
        ]}
        caption={
          <>
            <code>ImportanceLevel</code> is <code>LOW</code> (1), <code>MEDIUM</code> (2),{' '}
            <code>HIGH</code> (3) or <code>CRITICAL</code> (4), and decides what survives
            consolidation.
          </>
        }
      />

      <h2>Vector memory</h2>
      <p>
        For recall by meaning: the entry that answers the question, not the one that repeats its
        words. Backed by FAISS or Chroma, with embeddings from whichever provider you give it.
      </p>

      <CodeBlock filename="vector.py" code={`from effgen.memory import VectorMemoryStore

store = VectorMemoryStore(backend_type="faiss", persist_directory="/tmp/effgen-vec")
store.add("The deploy runs at 02:00 UTC.")
store.add("Pixel is a border collie.")

for hit in store.search("when does the release go out?", k=1):
    print(hit.rank, round(hit.similarity, 3), "|", hit.entry.content)`} />

      <Terminal
        command="python vector.py"
        output={`0 0.414 | The deploy runs at 02:00 UTC.`}
        caption="The query shares no words with the entry it found. That is the difference from a keyword search."
      />

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'backend_type',
            type: 'str',
            default: "'faiss'",
            description: 'faiss or chroma. Each needs its own package installed.',
          },
          {
            name: 'embedding_provider',
            type: 'EmbeddingProvider | None',
            default: 'None',
            description: 'What turns text into vectors. Defaults to the bundled sentence-transformer embedding.',
          },
          {
            name: 'persist_directory',
            type: 'str | Path | None',
            default: 'None',
            description: 'Where the index is written. Without one it lives in memory only.',
          },
          {
            name: 'consolidation_threshold',
            type: 'int',
            default: '1000',
            description: 'How many entries before a consolidation pass.',
          },
          { name: 'max_entries', type: 'int', default: '10000', description: 'The cap on stored entries.' },
        ]}
        caption={<><code>effgen.memory.VectorMemoryStore</code></>}
      />

      <ApiTable
        headers={['Call', 'What it does']}
        rows={[
          [
            <code>add(content, entry_id=None, metadata=None)</code>,
            <>
              Embed and store one string. Note it takes text, not a <code>MemoryEntry</code>.
            </>,
          ],
          [<code>add_batch(items)</code>, 'The same for many, in one embedding pass.'],
          [
            <code>search(query, k=10, min_similarity=0.0)</code>,
            <>
              Returns <code>SearchResult</code> records — <code>entry</code>,{' '}
              <code>similarity</code>, <code>rank</code>.
            </>,
          ],
          [<>
            <code>get(entry_id)</code>, <code>delete(entry_id)</code>
          </>, 'One entry by id.'],
          [<>
            <code>save()</code>, <code>load()</code>
          </>, 'Write the index to persist_directory, or read it back.'],
          [<code>consolidate()</code>, 'Drop near-duplicates.'],
          [<code>get_statistics()</code>, 'Entry count, dimension and backend.'],
        ]}
      />

      <Callout type="tip" title="For documents, use RAG rather than vector memory">
        <p>
          Vector memory is for things the agent should remember. Indexing a corpus you want it to
          answer from is <Link to="/rag">RAG</Link>, which adds chunking, hybrid search, reranking
          and citations back to the source.
        </p>
      </Callout>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'The agent does not remember the previous turn',
            <>
              <code>enable_memory</code> is off, or each turn built a new agent.
            </>,
            <>
              Set <code>enable_memory=True</code> and reuse the agent, or pass a{' '}
              <Link to="/sessions">session</Link> so the history survives the process.
            </>,
          ],
          [
            'Old turns have quietly become a summary',
            'The history passed the compaction threshold, which is what is supposed to happen.',
            <>
              <code>get_statistics()["total_summarizations"]</code> says how often. Choose what is
              dropped with a <Link to="/compaction">compaction strategy</Link>.
            </>,
          ],
          [
            <><code>TypeError: add_memory() missing 1 required positional argument: 'memory_type'</code></>,
            <>
              A <code>MemoryEntry</code> was passed where the arguments were expected.
            </>,
            <>
              <code>add_memory(content=…, memory_type=…)</code>. The entry is what comes back, not
              what goes in.
            </>,
          ],
          [
            <><code>ValueError: Unsupported input type: MemoryEntry</code></>,
            <>
              A <code>MemoryEntry</code> was handed to <code>VectorMemoryStore.add()</code>, which
              takes text.
            </>,
            <>
              <code>store.add("the text")</code>. The embedder needs a string.
            </>,
          ],
          [
            <><code>ImportError</code> naming faiss or chromadb</>,
            'The vector backend is not installed.',
            <>
              <code>pip install effgen[rag]</code>, or install the backend you asked for.
            </>,
          ],
          [
            'A locked SQLite file',
            <>
              A <code>LongTermMemory</code> was left open.
            </>,
            <>
              Call <code>close()</code>, or open the store in a <code>with</code> block.
            </>,
          ],
          [
            'A token count that does not match the provider’s',
            'The history was estimated at four characters per token.',
            <>
              Pass <code>tokenizer=</code> — anything with <code>count_tokens</code> or{' '}
              <code>encode</code>. One that raises falls back to the estimate rather than failing
              the run.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/sessions', '/compaction', '/rag']} />
    </DocPage>
  );
}
