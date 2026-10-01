import { History } from 'lucide-react';
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

export default function CliHistory() {
  return (
    <DocPage
      subtitle="Two stores: every run that finished, and every conversation you named. What each keeps, and how to search it."
      icon={<History size={48} />}
    >
      <p>
        effGen writes down what it did. <code>effgen runs</code> reads the record of finished runs —
        one row per run, whatever started it — and <code>effgen sessions</code> reads the
        conversations, which exist only for runs you gave a <code>--session-id</code>. They are
        separate stores answering separate questions: <em>what has this machine run?</em> and{' '}
        <em>what did we say to each other?</em>
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen runs list                  # what has run here
effgen sessions list              # what conversations are saved`}
      />

      <Terminal
        command="effgen runs list"
        output={`
Run           When              Model                     Task                                           Cost  Time  Status
------------  ----------------  ------------------------  ----------------------------------------  ---------  ----  ------
22a1d08a3449  2026-08-24 18:58  openai:gpt-5-nano         What is 18723 * 4409? Use the calculato…    $0.0002  3.8s  ok
fdc5d776daa1  2026-08-24 18:58  groq:openai/gpt-oss-20b   Say ok.                                   $0.000029  0.3s  ok
6ca865821efe  2026-08-24 18:58  openai:gpt-5-nano         What is 7*6? Use the calculator.          $0.000048  1.9s  ok
cabdeb290b4e  2026-08-24 18:57  openai:gpt-5-nano         What is 2+2?                              $0.000040  3.0s  ok
460cb5fe0ed0  2026-08-24 18:57  gemini:gemini-3.1-flash…  Say the word ok and nothing else.         $0.000039  2.4s  ok
e4fbcf737fcd  2026-08-24 18:57  groq:llama-3.1-8b-insta…  What is 18723 * 4409? Use the calculato…          —  0.1s  error
Stored in: /data/wang/gks/effgen-d6/home/runs
Open one with: effgen runs show <run-id>`}
        caption={
          <>
            A real store, on the machine this page was written on. The failed row cost nothing
            because it never reached the model, and a run with no published price would read{' '}
            <code>—</code> rather than <code>$0</code>.
          </>
        }
      />

      <h2>The two stores</h2>

      <ApiTable
        headers={['What is compared', 'effgen runs', 'effgen sessions']}
        rows={[
          [
            'What one row is',
            'One finished run.',
            'One named conversation, however many turns long.',
          ],
          [
            'How it is created',
            'Automatically, by every run that completes — the CLI, a script, a team, the server.',
            <>
              Only when a run names one: <code>--session-id</code>, or an interactive{' '}
              <code>effgen chat</code>.
            </>,
          ],
          [
            'What it keeps',
            'Model, provider, task, a truncated answer, tokens, cost, duration, status, and the execution and session ids.',
            'Every message in order, each with its model, tokens, cost and latency.',
          ],
          [
            'Where',
            <code>$EFFGEN_HOME/runs</code>,
            <code>$EFFGEN_HOME/sessions</code>,
          ],
          [
            'On disk as',
            'One JSONL file per day, appended.',
            'One document per session id.',
          ],
          [
            'What continues it',
            <>
              Nothing — a run is finished. <code>effgen resume</code> restarts from a{' '}
              <Link to="/checkpointing">checkpoint</Link>, which is a third store again.
            </>,
            <>
              <code>effgen chat --session-id ID</code>, <code>effgen run … --session-id ID</code> or{' '}
              <code>effgen code --session-id ID</code>.
            </>,
          ],
          [
            'Default retention',
            <>
              30 days (<code>EFFGEN_RUN_HISTORY_MAX_DAYS</code>).
            </>,
            <>
              Kept until <code>sessions cleanup</code> or <code>sessions delete</code> removes them.
            </>,
          ],
        ]}
      />

      <h2>Searching runs</h2>

      <ParamTable
        nameLabel="Flag"
        params={[
          { name: '--json', type: 'flag', description: 'Output as JSON' },
          {
            name: '--status {ok,stopped,error,failed}',
            description: (
              <>
                Only runs with this status (<code>failed</code> is an alias for <code>error</code>;{' '}
                <code>stopped</code> is a run the loop ended before the model wrote an answer)
              </>
            ),
          },
          { name: '-m MODEL, --model MODEL', description: 'Only runs on a model matching this text' },
          {
            name: '--search SEARCH',
            description: 'Only runs whose task, answer, id or error match this text',
          },
          {
            name: '--session-id SESSION_FILTER',
            description: 'Only runs from this session',
          },
          { name: '--since SINCE', description: 'Only runs on/after this date (YYYY-MM-DD)' },
          { name: '--until UNTIL', description: 'Only runs on/before this date (YYYY-MM-DD)' },
          { name: '--limit LIMIT', default: '20', description: 'Maximum runs to show' },
        ]}
        caption={
          <>
            Every flag <code>effgen runs list --help</code> declares. The text filters are
            substring matches, not globs.
          </>
        }
      />

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen runs list --status error
effgen runs list --model gpt-5-nano --limit 3
effgen runs list --since 2026-08-01 --search calculator`}
      />

      <Terminal
        command="effgen runs list --status error"
        output={`
Run           When              Model                     Task                                      Cost  Time  Status
------------  ----------------  ------------------------  ----------------------------------------  ----  ----  ------
e4fbcf737fcd  2026-08-24 18:57  groq:llama-3.1-8b-insta…  What is 18723 * 4409? Use the calculato…     —  0.1s  error
Stored in: /data/wang/gks/effgen-d6/home/runs
Open one with: effgen runs show <run-id>`}
      />

      <h2>Reading one run</h2>

      <CodeBlock language="bash" filename="terminal" code={`effgen runs show 34fa69a237d8`} />

      <Terminal
        command="effgen runs show 34fa69a237d8"
        output={`Run:      34fa69a237d8
When:     2026-08-24T18:59:25-04:00
Status:   ok
Model:    openai:gpt-5-nano (openai)
Agent:    cli-agent
Tokens:   219 in / 153 out
Cost:     $0.000072
Duration: 1.36s

Task:
What is 7*6? Use the calculator.

Answer:
42`}
      />

      <ParamTable
        nameLabel="Flag"
        params={[
          { name: 'run_id', required: true, description: 'Run id, as shown by `effgen runs list`' },
          { name: '--json', type: 'flag', description: 'Output as JSON' },
          {
            name: '--card PATH.html',
            type: 'path',
            description: (
              <>
                Write a summary HTML card for this stored run to PATH. History keeps a truncated
                answer and no step trace, so the card states that; use <code>effgen run --card</code>{' '}
                at run time for the full answer, trace and sources.
              </>
            ),
          },
        ]}
        caption={
          <>
            From <code>effgen runs show --help</code>. <Link to="/cli/reports">Reports and run
            cards</Link> covers what a card contains.
          </>
        }
      />

      <h3>The stored fields</h3>

      <Terminal
        command={`effgen runs list --json | jq ".runs[0] | keys"`}
        output={`[
  "agent",
  "cost_usd",
  "duration_s",
  "error",
  "execution_id",
  "execution_kind",
  "execution_name",
  "input_tokens",
  "model",
  "output",
  "output_tokens",
  "parent_agent",
  "provider",
  "role",
  "run_id",
  "session_id",
  "status",
  "task",
  "ts"
]`}
        maxLines={22}
        caption={
          <>
            The <code>execution_*</code>, <code>parent_agent</code> and <code>role</code> fields are
            what group a <Link to="/multi-agent">team</Link> or a{' '}
            <Link to="/workflows">workflow</Link>'s runs together: every member of one execution
            carries the same <code>execution_id</code>.
          </>
        }
      />

      <Callout type="tip" title="One run, many rows">
        <p>
          A team run writes a row per member agent, not one row for the team. Filter with{' '}
          <code>jq 'group_by(.execution_id)'</code>, or read the same grouping already done for you
          in the dashboard's <Link to="/dashboard">topology panel</Link>.
        </p>
      </Callout>

      <h2>Conversations</h2>

      <p>
        A session is created the first time a run names one. The id is yours to choose — it is how
        you find the conversation later, so a meaningful one is worth more than a generated one.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen run "My favourite number is 41. Remember it." -m openai:gpt-5-nano --session-id demo-42 -q
effgen run "What is my favourite number plus one?" -m openai:gpt-5-nano --session-id demo-42 -q`}
      />

      <Terminal
        command={`effgen run "What is my favourite number plus one?" -m openai:gpt-5-nano --session-id demo-42 -q`}
        output={`
Response
╭───────────────────────────────────────── Agent Response ─────────────────────────────────────────╮
│ 42                                                                                               │
│                                                                                                  │
│ Since your favourite number is 41, 41 + 1 = 42.                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯`}
        caption="Two separate processes, minutes apart. The second one recalled the first because they named the same session."
      />

      <Terminal
        command="effgen sessions list"
        output={`#  Session  Messages  Model                 Cost  Updated
-  -------  --------  -----------------  -------  ----------------
1  demo-42         4  openai:gpt-5-nano  $0.0003  2026-08-24 19:00
Stored in: /data/wang/gks/effgen-d6/home/sessions
Read one:  effgen sessions show <id>
Continue:  effgen chat --session-id <id>`}
      />

      <h3>Reading a conversation</h3>

      <Terminal
        command="effgen sessions show demo-42 --last 4"
        output={`Session: demo-42
Agent:   cli-agent
Model:   openai:gpt-5-nano
Turns:   4 message(s)

--- user · 2026-08-24 19:00  ·  openai:gpt-5-nano  652 tok  $0.0002  3206 ms
My favourite number is 41. Remember it.

--- assistant · 2026-08-24 19:00  ·  openai:gpt-5-nano  652 tok  $0.0002  3206 ms
Got it. Your favourite number is 41. I'll remember it for the duration of this chat. How would you like me to use it—e.g., include it in example problems or puzzles?

--- user · 2026-08-24 19:00  ·  openai:gpt-5-nano  510 tok  $0.0001  2584 ms
What is my favourite number plus one?

--- assistant · 2026-08-24 19:00  ·  openai:gpt-5-nano  510 tok  $0.0001  2584 ms
42

Since your favourite number is 41, 41 + 1 = 42.

Continue this conversation: effgen chat --session-id demo-42`}
        maxLines={22}
        caption="The tokens, cost and latency on a pair of messages are the turn's, so the same figures appear on the question and the answer that closed it."
      />

      <h2>The sessions sub-commands</h2>

      <ApiTable
        headers={['Sub-command', 'Options', 'Does']}
        rows={[
          [
            <code>sessions list</code>,
            <>
              <code>--json</code> · <code>--search</code> · <code>--since</code> ·{' '}
              <code>--until</code> · <code>-m/--model</code> · <code>--limit</code> (50)
            </>,
            'Every saved conversation, most recently updated first.',
          ],
          [
            <code>sessions show &lt;id&gt;</code>,
            <>
              <code>--last N</code> · <code>--json</code>
            </>,
            'The conversation, turn by turn.',
          ],
          [
            <code>sessions browse</code>,
            <>
              <code>--json</code> · <code>--limit</code> (20)
            </>,
            'Pick one from a numbered list and read it. On a terminal only — with --json it prints the list instead of prompting.',
          ],
          [
            <code>sessions export &lt;id&gt;</code>,
            <>
              <code>--format {'{'}json,text{'}'}</code> (json)
            </>,
            'Write the conversation to stdout, as data or as a transcript.',
          ],
          [<code>sessions delete &lt;id&gt;</code>, '—', 'Remove one session.'],
          [
            <code>sessions cleanup</code>,
            <>
              <code>--days N</code> (30)
            </>,
            'Remove sessions older than N days.',
          ],
        ]}
        caption={
          <>
            From each sub-command's <code>--help</code>. <code>runs</code> has three:{' '}
            <code>list</code>, <code>show</code> and <code>cleanup --days N</code>.
          </>
        }
      />

      <Terminal
        command="effgen sessions export demo-42 --format text"
        output={`Session: demo-42
Agent: cli-agent

[user] My favourite number is 41. Remember it.
[assistant] Got it. Your favourite number is 41. I'll remember it for the duration of this chat. How would you like me to use it—e.g., include it in example problems or puzzles?
[user] What is my favourite number plus one?
[assistant] 42

Since your favourite number is 41, 41 + 1 = 42.`}
      />

      <h2>Turning history off, and cleaning it up</h2>

      <ApiTable
        headers={['Variable', 'Effect']}
        rows={[
          [
            <code>EFFGEN_HOME</code>,
            <>
              Where both stores live. Default <code>~/.effgen</code>. Set it per project to keep
              histories apart; a container needs it on a mounted volume or the record dies with the
              container.
            </>,
          ],
          [
            <code>EFFGEN_RUN_HISTORY=0</code>,
            'Keep run history in memory only — nothing is written to disk. Sessions are unaffected.',
          ],
          [
            <code>EFFGEN_RUN_HISTORY_MAX_DAYS</code>,
            <>
              Retention for the run store, in days. Default <code>30</code>.
            </>,
          ],
        ]}
      />

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen runs cleanup --days 7        # drop run history older than a week
effgen sessions cleanup --days 90   # drop conversations older than a quarter`}
        caption="Both delete files. Neither asks first, so check the window before running it in a shared home."
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>No runs recorded yet.</code> after a run that clearly happened
            </>,
            <>
              A different <code>EFFGEN_HOME</code> was in effect, or{' '}
              <code>EFFGEN_RUN_HISTORY=0</code> kept it in memory.
            </>,
            <>
              <code>effgen runs list</code> prints the directory it read at the bottom of the
              table — compare it with the one the run used.
            </>,
          ],
          [
            <>
              <code>No sessions yet.</code> after several runs
            </>,
            'Expected. A session exists only where a run named one; runs alone create no conversation.',
            <>
              Add <code>--session-id</code>, or start an <code>effgen chat</code>.
            </>,
          ],
          [
            'A session recalls nothing from the previous turn',
            'A new id was typed, or the id differs by a character. Ids are exact.',
            <>
              <code>effgen sessions list</code> shows what exists;{' '}
              <code>effgen sessions list --search &lt;text&gt;</code> finds one by content.
            </>,
          ],
          [
            'A run is missing from the list but its answer arrived',
            'Only completed runs are recorded. A run still executing, or one killed mid-flight, has nothing to record.',
            <>
              For a long run that may not finish, checkpoint it —{' '}
              <Link to="/checkpointing">checkpointing</Link> — and resume from the snapshot.
            </>,
          ],
          [
            <>
              <code>effgen resume --session-id …</code> is rejected
            </>,
            <>
              <code>resume</code> takes <code>--checkpoint</code>, not a session. They are different
              stores.
            </>,
            <>
              To continue a conversation use <code>effgen chat --session-id ID</code>. To restart an
              interrupted run use <code>effgen resume --checkpoint ID</code>.
            </>,
          ],
          [
            'The store grows without bound',
            <>
              Retention prunes runs on read, not sessions —{' '}
              <code>sessions cleanup</code> is manual.
            </>,
            <>
              Run <code>effgen sessions cleanup --days N</code> on a schedule, or lower{' '}
              <code>EFFGEN_RUN_HISTORY_MAX_DAYS</code>.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          Both commands are new. Earlier releases kept no durable record of a run and no way to
          continue a conversation by name; <code>--session-id</code> on <code>run</code>,{' '}
          <code>chat</code> and <code>code</code> writes into the one store all three read.
        </p>
      </Callout>

      <SeeAlso paths={['/sessions', '/cli/reports', '/checkpointing']} />
    </DocPage>
  );
}
