import { Scissors } from 'lucide-react';
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

export default function Compaction() {
  return (
    <DocPage
      subtitle="What happens when a conversation outgrows the context window, and how to choose what is dropped."
      icon={<Scissors size={48} />}
    >
      <p>
        A long conversation eventually does not fit, and something has to go. Which turns survive
        changes the answer more for a small model than for a frontier one, and different tasks want
        different answers — so in effGen the choice is a strategy you pass, not a fixed rule.
      </p>

      <Callout type="note" title="Two kinds of compaction">
        <p>
          This page has two parts. A session’s <strong>history</strong> is compacted by the memory
          strategy below, which is new in 1.0.0. Since 1.1.0 a single <strong>run</strong> is also
          bounded by the prompt tokens it may send, and compacts its own thread to stay inside
          them — <a href="#run-budget">keeping a run inside its budget</a>.
        </p>
      </Callout>

      <h2>It already happens</h2>
      <p>
        Compaction is not something you switch on. A <code>ShortTermMemory</code> compacts itself
        once the history passes a fraction of its window, and says so.
      </p>

      <CodeBlock filename="growing.py" code={`from effgen.memory import ShortTermMemory

# A 512-token window with the last four turns pinned. The memory compacts
# itself as the history grows past the threshold.
memory = ShortTermMemory(max_tokens=512, keep_recent_messages=4, summarization_threshold=0.8)
for i in range(14):
    memory.add_user_message(f"Turn {i}: " + "context that has to go somewhere. " * 4)

stats = memory.get_statistics()
print("messages added   ", stats["total_messages_added"])
print("messages held    ", stats["current_messages"])
print("summarizations   ", stats["total_summarizations"])
print("summaries kept   ", stats["summaries_count"])
print("tokens           ", stats["current_tokens"], "of", stats["max_tokens"])`} />

      <Terminal
        command="python growing.py"
        output={`messages added    14
messages held     6
summarizations    1
summaries kept    1
tokens            303 of 512`}
        caption={`Run against effGen 1.0.0. Fourteen turns went in; six are held, because one compaction pass replaced the older ones with a summary.`}
      />

      <p>
        The default is <code>SummarizeOldest</code>: past{' '}
        <code>summarization_threshold</code> × the window, everything but the most recent few turns
        becomes a summary.
      </p>

      <h2>Choosing a different one</h2>

      <CodeBlock filename="configure.py" code={`from effgen import Agent, AgentConfig
from effgen.memory.compaction import KeepFirstAndLast

agent = Agent(AgentConfig(
    model="gpt-5-nano",
    provider="openai",
    compaction_strategy=KeepFirstAndLast(first=2, last=6),
))
print(type(agent.config.compaction_strategy).__name__)

named = Agent(AgentConfig(model="gpt-5-nano", provider="openai", compaction_strategy="drop_oldest"))
print(named.config.compaction_strategy)`} />

      <Terminal command="python configure.py" output={`KeepFirstAndLast
drop_oldest`} />

      <p>
        A strategy can be an instance, so it can be configured, or a name, which is convenient in a
        config file.
      </p>

      <h2>What ships</h2>

      <ApiTable
        headers={['Strategy', 'What survives', 'Calls a model?']}
        rows={[
          [
            <><code>SummarizeOldest</code> <em>(default)</em></>,
            'The most recent turns verbatim; everything older becomes a summary.',
            'Yes',
          ],
          [<code>DropOldest</code>, 'The most recent turns. The rest is forgotten.', 'No'],
          [
            <code>KeepFirstAndLast</code>,
            'The opening turns and the recent ones; the middle is summarised or dropped.',
            'Optional',
          ],
          [
            <code>KeepToolResults</code>,
            'Tool results and the recent turns; older reasoning is compacted.',
            'Optional',
          ],
        ]}
        caption={<>All four are in <code>effgen.memory.compaction</code>.</>}
      />

      <CodeBlock filename="strategies.py" code={`from effgen.memory import ShortTermMemory
from effgen.memory.compaction import (
    DropOldest, KeepFirstAndLast, KeepToolResults, SummarizeOldest,
)


def conversation():
    """One transcript, rebuilt per strategy so each sees the same history."""
    memory = ShortTermMemory(max_tokens=100_000, keep_recent_messages=4)
    memory.add_user_message("Review this contract for termination risk.")
    memory.add_assistant_message("I will read clause 7 first.")
    for i in range(8):
        memory.add_assistant_message(f"Step {i}: reasoning about the termination window.")
    memory.add_tool_message("pdf → Clause 7.2: either party may terminate with 30 days notice.")
    memory.add_assistant_message("Clause 7.2 is the risk.")
    return memory


for strategy in (SummarizeOldest(), DropOldest(), KeepFirstAndLast(first=2, last=6), KeepToolResults()):
    memory = conversation()
    total = len(memory.get_messages())
    leaving = strategy.messages_to_compact(memory)
    print(f"{type(strategy).__name__:18} compacts {len(leaving):2} of {total}, keeps {total - len(leaving)}")`} />

      <Terminal
        command="python strategies.py"
        output={`SummarizeOldest    compacts  8 of 12, keeps 4
DropOldest         compacts  8 of 12, keeps 4
KeepFirstAndLast   compacts  4 of 12, keeps 8
KeepToolResults    compacts  8 of 12, keeps 4`}
        caption="The same twelve-message transcript, put to each strategy. KeepFirstAndLast keeps twice as much, because it holds on to the opening turns as well as the recent ones."
      />

      <ApiTable
        headers={['Reach for', 'When']}
        rows={[
          [
            <code>DropOldest</code>,
            'Older turns genuinely do not matter. It costs nothing, waits for nothing, and cannot invent anything — a summary is model output, with everything that implies.',
          ],
          [
            <code>KeepFirstAndLast</code>,
            'The opening turns carry the task — the document, the instruction, the constraint everything else refers to — and a summary of those is a poor substitute. The redundancy in a long conversation is in the middle.',
          ],
          [
            <code>KeepToolResults</code>,
            'A tool-heavy run, where the reasoning is most of the tokens and the tool results are the evidence the answer rests on.',
          ],
          [
            <code>SummarizeOldest</code>,
            'A general conversation where the gist of the older turns matters more than any particular one, and the extra model call is worth it.',
          ],
        ]}
      />

      <h2>Measuring the history</h2>
      <p>
        By default the history is measured with the model's own tokenizer when there is one, and
        otherwise estimated at four characters per token. Supply a tokenizer to measure it in the
        units the window is actually counted in.
      </p>

      <CodeBlock filename="tokenizer.py" code={`import tiktoken

from effgen.memory import ShortTermMemory

estimated = ShortTermMemory()
measured = ShortTermMemory(tokenizer=tiktoken.get_encoding("cl100k_base"))
for memory in (estimated, measured):
    memory.add_user_message("Compaction decides which turns survive a long conversation.")

print("four-characters-per-token estimate:", estimated.get_token_count())
print("cl100k_base                       :", measured.get_token_count())`} />

      <Terminal
        command="python tokenizer.py"
        output={`four-characters-per-token estimate: 14
cl100k_base                       : 14`}
        caption="For this sentence the estimate and the real count agree. On code, on JSON and on a language that is not English, they do not."
      />

      <p>
        Anything with <code>count_tokens(text)</code> or <code>encode(text)</code> works. A
        tokenizer that raises falls back to the estimate rather than failing the run.
      </p>

      <h2>Writing your own</h2>
      <p>
        Subclass <code>CompactionStrategy</code> and override only what differs; the defaults are{' '}
        <code>SummarizeOldest</code>'s.
      </p>

      <CodeBlock filename="custom.py" code={`from effgen.memory import ShortTermMemory
from effgen.memory.compaction import CompactionStrategy


class DropFailedToolTurns(CompactionStrategy):
    """Compact the turns where a tool errored; keep everything else."""

    def messages_to_compact(self, memory):
        return [m for m in memory.messages if "Error executing tool" in m.content]

    def summarize(self, memory, messages):
        return None          # drop them rather than summarize them


memory = ShortTermMemory()
memory.add_user_message("Fetch the weather for Oslo.")
memory.add_tool_message("Error executing tool weather: connection refused")
memory.add_assistant_message("I could not reach the weather service.")

strategy = DropFailedToolTurns()
leaving = strategy.messages_to_compact(memory)
print(len(leaving), "message(s) compacted:", [m.content[:40] for m in leaving])
print("replacement:", strategy.summarize(memory, leaving))`} />

      <Terminal command="python custom.py" output={`1 message(s) compacted: ['Error executing tool weather: connection']
replacement: None`} />

      <ApiTable
        headers={['Method', 'Answers', 'Default']}
        rows={[
          [
            <code>should_compact(memory)</code>,
            'Is it time?',
            <>
              Past <code>summarization_threshold</code> × the window.
            </>,
          ],
          [
            <code>messages_to_compact(memory)</code>,
            'Which messages leave?',
            'Everything but the recent few.',
          ],
          [
            <code>summarize(memory, messages)</code>,
            'What replaces them?',
            <>
              A generated summary. <code>None</code> drops them instead.
            </>,
          ],
        ]}
        caption={
          <>
            Called in that order. Returning an empty list from{' '}
            <code>messages_to_compact</code> cancels that round.
          </>
        }
      />

      <h2 id="run-budget">Keeping a run inside its budget</h2>
      <p>
        <code>AgentConfig.context_budget</code> says how many prompt tokens a run may send. At the
        default, <code>"auto"</code>, the budget is the model’s window less the output reserve,
        times 0.85; it is unbounded when the model declares no window, because guessing one would
        truncate a conversation that would have fitted. An integer names the budget yourself,{' '}
        <code>None</code> leaves the run unbounded as in 1.0.x, and{' '}
        <code>max_context_length</code> overrides the window the model declares.{' '}
        <code>response.metadata["context_budget"]</code> reports it on every outcome.
      </p>

      <CodeBlock filename="budget.py" code={`from effgen import AgentConfig, ShortenOldestFirst

config = AgentConfig(
    model="openai:gpt-5-nano",
    context_budget=8000,
    compaction=ShortenOldestFirst(observation_keep_chars=120),
)
print(config.context_budget, type(config.compaction).__name__)`} />

      <Terminal command="python budget.py" output="8000 ShortenOldestFirst" caption="Run against effGen 1.1.0." />

      <p>
        When the conversation will not fit, <code>AgentConfig.compaction</code> brings the thread
        back under. The default, <code>ShortenOldestFirst</code>, makes no model call and gives up
        the oldest material first: an old tool result is shortened, then an old thought dropped,
        then whole answered cycles replaced by one <code>NudgeStep</code>. It never touches the
        frame, the task, the most recent two complete cycles or the answer, and a call and the
        result answering it always leave together. A shortened <code>ObservationStep</code> says so
        through <code>compacted</code> and <code>original_chars</code>.{' '}
        <code>SummarizeWithModel</code> is opt-in and leaves a model-written summary behind. A run
        that still cannot fit raises <code>ContextBudgetExceededError</code>, and a prompt larger
        than the model’s window is sent once rather than retried.
      </p>

      <h2>The settings that shape it</h2>

      <ParamTable
        nameLabel="Argument"
        params={[
          {
            name: 'max_tokens',
            type: 'int',
            default: '4096',
            description: 'The window the history is measured against.',
          },
          {
            name: 'summarization_threshold',
            type: 'float',
            default: '0.8',
            description: 'The fraction of the window at which compaction is triggered.',
          },
          {
            name: 'keep_recent_messages',
            type: 'int',
            default: '10',
            description: 'How many recent turns are never compacted.',
          },
          {
            name: 'summary_length_ratio',
            type: 'float',
            default: '0.3',
            description: 'How long a summary may be, relative to what it replaces.',
          },
          {
            name: 'summary_budget_ratio',
            type: 'float',
            default: '0.4',
            description: 'The share of the window summaries may occupy in total.',
          },
          {
            name: 'model',
            type: 'Any',
            default: 'None',
            description: 'The model that writes summaries. A strategy that needs one and has none cannot summarise.',
          },
          {
            name: 'tokenizer',
            type: 'Any',
            default: 'None',
            description: 'How the history is measured. Falls back to four characters per token.',
          },
        ]}
        caption={
          <>
            All on <code>ShortTermMemory</code>, and reachable through{' '}
            <code>AgentConfig</code> — see <Link to="/memory">Memory</Link>.
          </>
        }
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          Pluggable compaction is new in 1.0.0. Before it, what effGen did was what{' '}
          <code>SummarizeOldest</code> now does, and there was no way to change it. Code that does
          not pass <code>compaction_strategy</code> behaves exactly as it did.
        </p>
      </Callout>

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'The agent forgot something from earlier in the conversation',
            'That turn was compacted, and the summary did not carry it.',
            <>
              <code>KeepFirstAndLast</code> if it was in the opening turns,{' '}
              <code>KeepToolResults</code> if it was a tool result, or raise{' '}
              <code>keep_recent_messages</code>.
            </>,
          ],
          [
            'Compaction never happens on a long conversation',
            <>
              <code>max_tokens</code> is larger than the history ever gets, or the history is
              being measured at four characters per token and so looks smaller than it is.
            </>,
            <>
              <code>get_statistics()["total_summarizations"]</code> is the count. Pass a real{' '}
              <code>tokenizer</code>.
            </>,
          ],
          [
            'The summary contains something the conversation did not say',
            'A summary is model output.',
            <>
              Use <code>DropOldest</code> where invention is worse than forgetting. It calls no
              model.
            </>,
          ],
          [
            'A pause partway through a long conversation',
            'A compaction pass is generating a summary, which is a model call.',
            <>
              <code>DropOldest</code> or <code>KeepFirstAndLast</code> without summarisation costs
              nothing and waits for nothing.
            </>,
          ],
          [
            'A custom strategy never fires',
            <>
              <code>should_compact</code> was overridden and returns <code>False</code>, or{' '}
              <code>messages_to_compact</code> returns an empty list.
            </>,
            'An empty list cancels the round by design. Check both before looking elsewhere.',
          ],
          [
            'Context-length errors from the provider anyway',
            'The window the memory is sized to is larger than the model’s real one.',
            <>
              Set <code>max_tokens</code> from the model’s context length —{' '}
              <Link to="/catalog">the catalog</Link> lists it — and leave room for the answer.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/memory', '/sessions', '/agents']} />
    </DocPage>
  );
}
