import { MessagesSquare } from 'lucide-react';
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

export default function Sessions() {
  return (
    <DocPage
      subtitle="Carrying a conversation across runs with run(session=...), and where it is kept."
      icon={<MessagesSquare size={48} />}
    >
      <p>
        An agent's own memory ends with the process. A session is the same conversation written to
        disk under a name, so a later run — a later process, a later day, a different machine
        sharing the directory — picks it up where it stopped.
      </p>

      <h2>Two runs, one conversation</h2>

      <CodeBlock filename="session.py" code={`from effgen import Agent, AgentConfig

# The same session id in two separate agents — the second remembers the first.
first = Agent(AgentConfig(model="gpt-5-nano", provider="openai"), session_id="pets-demo")
first.run("My dog is named Pixel.")

second = Agent(AgentConfig(model="gpt-5-nano", provider="openai"), session_id="pets-demo")
print(second.run("What is my dog's name?").text)`} />

      <Terminal
        command="python session.py"
        output={`Pixel.`}
        caption={`Run against effGen 1.0.0. The second agent was constructed after the first had finished.`}
      />

      <p>
        A preset takes the same argument: <code>create_agent("math", "gpt-5-nano", session_id="user-123")</code>.
      </p>

      <h2>One agent, many conversations</h2>
      <p>
        <code>session_id=</code> on the constructor binds one conversation to the agent for its
        whole life: every <code>run()</code> and, since 1.3.0, every <code>stream()</code> turn is
        appended to it and saved. A server answering many people wants the opposite — one agent, and the
        conversation named per call. That is <code>session=</code> on <code>run()</code>.
      </p>

      <CodeBlock filename="per_call.py" code={`from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))

agent.run("My dog is named Pixel.", session="user-123")
agent.run("My cat is named Mote.", session="user-456")

print(agent.run("What is my pet's name?", session="user-123").text)
print(agent.run("What is my pet's name?", session="user-456").text)`} />

      <Terminal command="python per_call.py" output={`Pixel
Mote.`} />

      <Callout type="note" title="What run(session=…) does to the agent">
        <p>
          The run builds its prompt from that conversation's history and appends the turn to it.
          The two conversations never see each other, and the agent's own memory is left untouched
          — including when the run fails or a guardrail blocks the input. Without{' '}
          <code>session=</code>, <code>run()</code> uses the agent's own memory as it always did.
        </p>
      </Callout>

      <h3>Overlapping calls</h3>
      <p>
        Since 1.3.0 the conversation belongs to the call, not to the agent object, so the calls may
        overlap: threads calling <code>run()</code> on one agent, <code>run_async()</code> tasks on
        one event loop, and streams each read and record only their own session. In 1.2.0
        overlapping calls swapped the agent’s session on the shared object and could read and
        record each other’s turns. Inside a call, <code>agent.session</code> and{' '}
        <code>agent.short_term_memory</code> name that call’s conversation; outside one, they are the
        agent’s own. Per-call middleware applies to that call only.
      </p>

      <CodeBlock filename="overlapping.py" code={`import asyncio
from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))

async def main():
    await asyncio.gather(
        agent.run_async("What is my pet's name?", session="user-123"),
        agent.run_async("What is my pet's name?", session="user-456"),
    )

asyncio.run(main())`} />

      <p>
        <code>stream()</code> takes <code>session=</code> the same way: the streamed answer is
        appended to that conversation once the stream finishes. In 1.2.0 it ignored{' '}
        <code>session=</code> and wrote the turn to the agent’s own memory.
      </p>

      <CodeBlock continues filename="overlapping.py" code={`for chunk in agent.stream("And how old is he?", session="user-123"):
    print(chunk, end="")`} />

      <p>
        Calls made without <code>session=</code> share the agent’s own memory, so overlapping calls
        of that kind see each other’s turns — give each concurrent conversation a session.{' '}
        <code>agent.last_stream_response</code> and <code>agent.last_stream_usage</code> describe the
        stream that finished last; when streams overlap, read each one’s own record from its
        terminal <code>usage</code> event (<code>include_events=True</code>). Open at the edges in
        1.3.0: <code>stream()</code> runs no middleware, and <code>resume(session=...)</code>{' '}
        restores the checkpoint’s memory onto the agent’s own memory.
      </p>

      <h3>Passing the object instead of the id</h3>

      <CodeBlock filename="session_object.py" code={`from effgen import Agent, AgentConfig
from effgen.core.session import Session

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))
session = Session.load_or_create("pets-demo")

response = agent.run("And what breed is he?", session=session)
print(response.text)
print(len(session.messages), "messages in", session.session_id)`} />

      <Terminal command="python session_object.py" output={`The breed isn’t specified in the conversation.
6 messages in pets-demo`} />

      <h3>How the earlier turns reach the model</h3>
      <p>
        Since 1.1.0, a request that carries one string renders the session’s earlier turns as{' '}
        <code>Earlier in this conversation:</code> followed by <code>User:</code> /{' '}
        <code>Assistant:</code> lines. On a model whose adapter takes a conversation, a run whose
        tools travel as a request parameter sends them as their own <code>user</code> and{' '}
        <code>assistant</code> messages instead, and at the default{' '}
        <code>prompt_protocol="auto"</code> a run with tools that continues a session sends its own
        steps as messages too. <code>AgentConfig(prompt_protocol="flat")</code> keeps the run’s own
        steps in one string.
      </p>
      <p>
        <code>session.last_thread()</code> returns the steps of the most recent turn. An earlier
        turn’s stored steps are reduced to their shape unless the session is created with{' '}
        <code>keep_thread_history=True</code> — the default is <code>False</code>, which keeps
        session files from growing with every turn.
      </p>

      <h2>From the command line</h2>
      <p>
        Every command that runs an agent takes <code>--session-id</code>.
      </p>

      <CodeBlock
        filename="session.sh"
        language="bash"
        code={`effgen run -m gpt-5-nano --provider openai --session-id colors-demo \\
  --system-prompt "Answer in one short sentence." "My favourite colour is teal." -q
effgen run -m gpt-5-nano --provider openai --session-id colors-demo \\
  --system-prompt "Answer in one short sentence." "What is my favourite colour?" -q
effgen sessions list --search colors-demo`}
      />

      <Terminal command="bash session.sh" output={`
Response
╭─────────────────────────────── Agent Response ───────────────────────────────╮
│ Nice choice—teal is a calming and vibrant color.                             │
╰──────────────────────────────────────────────────────────────────────────────╯

Response
╭─────────────────────────────── Agent Response ───────────────────────────────╮
│ Your favourite colour is teal.                                               │
╰──────────────────────────────────────────────────────────────────────────────╯
#  Session      Messages  Model          Cost  Updated
-  -----------  --------  ----------  -------  ----------------
1  colors-demo         4  gpt-5-nano  $0.0002  2026-08-23 15:04
Stored in: /tmp/effgen-home/.effgen/sessions
Read one:  effgen sessions show <id>
Continue:  effgen chat --session-id <id>`} />

      <Terminal command="effgen sessions show colors-demo --last 2" output={`Session: colors-demo
Agent:   cli-agent
Model:   gpt-5-nano
Turns:   4 message(s) (showing last 2)

--- user · 2026-08-23 15:04  ·  gpt-5-nano  403 tok  $0.000070  1028 ms
What is my favourite colour?

--- assistant · 2026-08-23 15:04  ·  gpt-5-nano  403 tok  $0.000070  1028 ms
Your favourite colour is teal.

Continue this conversation: effgen chat --session-id colors-demo`} />

      <p>
        Each turn is stamped with the model that answered it, its token count, its cost and its
        latency — which is what <code>sessions list</code> and <code>sessions show</code> report.
      </p>

      <ApiTable
        headers={['Command', 'What it does']}
        rows={[
          [<code>effgen sessions list</code>, 'Id, message count, model, cost and when it was last updated.'],
          [<code>effgen sessions show &lt;id&gt;</code>, <>The conversation turn by turn. <code>--last N</code> tails a long one.</>],
          [<code>effgen sessions browse</code>, 'Lists, then lets you pick one to read. Prints the list and stops when stdin is not a terminal.'],
          [
            <code>effgen sessions export &lt;id&gt;</code>,
            <>
              JSON by default — a documented, reloadable format. <code>--format text</code> for a
              readable transcript.
            </>,
          ],
          [<code>effgen sessions delete &lt;id&gt;</code>, 'Exit 0 if it was removed, 1 if there was no such id.'],
          [<code>effgen sessions cleanup --days 30</code>, 'Delete sessions not updated in N days.'],
        ]}
        caption={
          <>
            <code>list</code> narrows with <code>--search</code> (ids, agent names and message
            content), <code>--model</code>, <code>--since</code>/<code>--until</code> and{' '}
            <code>--limit</code>. <code>list</code>, <code>show</code> and <code>browse</code> all
            take <code>--json</code>.
          </>
        }
      />

      <Terminal command="effgen sessions export colors-demo --format text" output={`Session: colors-demo
Agent: cli-agent

[user] My favourite colour is teal.
[assistant] Nice choice—teal is a calming and vibrant color.
[user] What is my favourite colour?
[assistant] Your favourite colour is teal.`} />

      <p>
        Export writes to stdout byte for byte, so message content containing square brackets —{' '}
        <code>[bold]</code>, <code>[REDACTED]</code>, <code>[1]</code> — survives intact and the
        JSON form round-trips.
      </p>

      <h2>Where a session is kept</h2>

      <ParamTable
        nameLabel="Variable"
        params={[
          {
            name: 'EFFGEN_SESSIONS_DIR',
            description: 'The exact directory for session files. Read at call time, so a change needs no restart.',
          },
          {
            name: 'EFFGEN_HOME',
            default: '~/.effgen',
            description: 'The base state directory. Sessions go in $EFFGEN_HOME/sessions, run history in $EFFGEN_HOME/runs, and the cost and rate-limit databases beside them.',
          },
        ]}
        caption={
          <>
            By default one JSON file per session at{' '}
            <code>~/.effgen/sessions/&lt;id&gt;.json</code>.
          </>
        }
      />

      <Callout type="tip" title="Saves are atomic">
        <p>
          A session is written to a temporary file and renamed, so a crash mid-write cannot leave a
          truncated file that fails to load later. A file that genuinely cannot be parsed is named
          in the output rather than dropped from the count, and exits <code>2</code>.
        </p>
      </Callout>

      <p>
        To keep growth bounded, schedule <code>effgen sessions cleanup --days N</code> from cron, or
        call <code>SessionManager(...).cleanup(older_than_days=N)</code>.
      </p>

      <h2>Run history</h2>
      <p>
        Separately from sessions, every agent run is recorded — keyed by the same{' '}
        <code>run_id</code> its trace spans carry, so a run can be found from a trace and a
        conversation from a run.
      </p>

      <Terminal command="effgen runs list --limit 3" output={`Run           When              Model       Task                               Cost  Time  Status
------------  ----------------  ----------  ----------------------------  ---------  ----  ------
b33fe7640c86  2026-08-23 15:04  gpt-5-nano  What is my favourite colour?  $0.000070  1.0s  ok
a4bc7a8f615b  2026-08-23 15:04  gpt-5-nano  My favourite colour is teal.    $0.0001  1.9s  ok
Stored in: /tmp/effgen-home/.effgen/runs
Open one with: effgen runs show <run-id>`} />

      <ApiTable
        headers={['Command', 'What it does']}
        rows={[
          [<code>effgen runs list</code>, <>Run id, time, model, task, cost and status. Narrows with <code>--status</code>, <code>--search</code> and <code>--since</code>.</>],
          [<code>effgen runs show &lt;run_id&gt;</code>, 'Task, answer, tokens, cost, error, and the session it belonged to.'],
          [<code>effgen runs cleanup --days 30</code>, 'Drop history older than N days.'],
        ]}
      />

      <ParamTable
        nameLabel="Variable"
        params={[
          { name: 'EFFGEN_RUN_HISTORY_DIR', description: 'The exact directory for run-history files.' },
          {
            name: 'EFFGEN_RUN_HISTORY',
            default: '1',
            description: '0 keeps history in memory only — nothing is written to disk.',
          },
          {
            name: 'EFFGEN_RUN_HISTORY_MAX_DAYS',
            default: '30',
            description: 'Retention in days. Older files are removed on write.',
          },
        ]}
        caption="Records are appended to one JSONL file per day, so the CLI, a script and the server share one history and it survives a restart."
      />

      <p>
        History writes are best-effort: a store that cannot be written does not fail the run. Task
        and answer text is stored as a 500-character preview. The dashboard's History view reads
        the same store.
      </p>

      <h2>Running work off the main thread</h2>

      <CodeBlock filename="background.py" code={`from effgen import Agent, AgentConfig
from effgen.core.background import BackgroundTaskRunner

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))

with BackgroundTaskRunner(agent, max_workers=2) as runner:
    task_id = runner.submit("Name one prime number between 10 and 20.")
    result = runner.get_result(task_id, wait=True, timeout=60)
    print(runner.get_status(task_id))
    print(result.text)`} />

      <Terminal command="python background.py" output={`COMPLETED
11`} />

      <ApiTable
        headers={['Method', 'What it does']}
        rows={[
          [<code>submit(task)</code>, 'Queue work and return a task id immediately.'],
          [<code>get_status(task_id)</code>, 'PENDING, RUNNING, COMPLETED, FAILED or CANCELLED.'],
          [<code>get_result(task_id, wait=False, timeout=None)</code>, <>The <code>AgentResponse</code>; block for it with <code>wait=True</code>.</>],
          [<>
            <code>cancel(task_id)</code>, <code>pause(task_id)</code>, <code>resume(task_id)</code>
          </>, 'Control one queued or running task.'],
        ]}
        caption={
          <>
            A failing task is reported as <code>FAILED</code> with a typed, secret-redacted error —
            never a silent success. The runner stops its threads when it is closed, used as a
            context manager, garbage-collected, or at interpreter exit, so a forgotten{' '}
            <code>shutdown()</code> does not leak threads. Prefer the <code>with</code> form.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>Session not found: &lt;id&gt;</code>,
            'No session file with that id, in the directory currently configured.',
            <>
              <code>effgen sessions list</code> shows what is there and prints the directory it
              read. Check <code>EFFGEN_SESSIONS_DIR</code> and <code>EFFGEN_HOME</code>.
            </>,
          ],
          [
            'Exit code 2 from a sessions command',
            'A session file could not be parsed.',
            'The file is named in the output rather than dropped from the count. Move it aside or delete it.',
          ],
          [
            'The conversation is not carried between runs',
            <>
              <code>enable_memory</code> was set instead of a session, or the two runs used
              different ids.
            </>,
            <>
              Memory is per-process; a session is on disk. Pass the same{' '}
              <code>session_id</code> or the same <code>session=</code>.
            </>,
          ],
          [
            'Two users see each other’s history',
            <>
              One agent was built with <code>session_id=</code>, so every run shares that
              conversation.
            </>,
            <>
              Use <code>run(task, session=…)</code> per call instead. The two never mix, also when
              the calls overlap (since 1.3.0), and the agent’s own memory is left as it was.
            </>,
          ],
          [
            'A session grows until it is slow to load',
            'Nothing prunes it.',
            <>
              <code>effgen sessions cleanup --days N</code>, or a{' '}
              <Link to="/compaction">compaction strategy</Link> so the history is summarised as it
              grows.
            </>,
          ],
          [
            'Nothing in `effgen runs list`',
            <>
              <code>EFFGEN_RUN_HISTORY=0</code>, or the store could not be written.
            </>,
            'History is best-effort and never fails a run, so an unwritable directory is silent. Check the path is writable.',
          ],
          [
            <>A background task reported <code>FAILED</code></>,
            'The agent run raised.',
            <>
              <code>get_result()</code> carries the typed error, with secrets redacted. It is never
              reported as a success.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/memory', '/checkpointing', '/compaction']} />
    </DocPage>
  );
}
