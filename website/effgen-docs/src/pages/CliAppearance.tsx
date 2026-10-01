import { Palette } from 'lucide-react';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { siteData } from '../siteData';

const themes = siteData.cli.themes;

export default function CliAppearance() {
  return (
    <DocPage
      subtitle="How the command line decides what to draw — colour, themes, animation and the bare landing — and how to turn each of it off."
      icon={<Palette size={48} />}
    >
      <p>
        Every visual the <code>effgen</code> binary produces is for an interactive terminal and every
        one is opt-out. Piped output, a redirect, a CI log and <code>--json</code> are plain text and
        none of the settings below change them — which is the property that lets the same command
        appear in a demo and in a cron job.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen --theme high-contrast presets    # one command
export EFFGEN_THEME=light                # the whole session
NO_COLOR=1 effgen presets                # structure, no colour at all`}
      />

      <h2>The four themes</h2>

      <ApiTable
        headers={['Theme', 'Use it for']}
        rows={[
          [<code>{themes[0]}</code>, 'The standard look, unchanged from earlier releases.'],
          [
            <code>{themes[1]}</code>,
            'Bright, bold hues and no dim text — written for low-vision readers.',
          ],
          [
            <code>{themes[2]}</code>,
            'Structure only: bold and underline carry the meaning that hue usually carries.',
          ],
          [<code>{themes[3]}</code>, 'Tuned for a light terminal background.'],
        ]}
        caption={
          <>
            The list is <code>effgen --help</code>'s own:{' '}
            <code>--theme {'{'}{themes.join(',')}{'}'}</code>.
          </>
        }
      />

      <p>
        <code>--theme</code> may appear before or after the sub-command, so both of these are
        accepted:
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen --theme monochrome models list
effgen models list --theme monochrome`}
      />

      <Callout type="tip" title="Choosing a theme for a screenshot or a demo">
        <p>
          <code>monochrome</code> survives being pasted into a document that will be printed, and{' '}
          <code>high-contrast</code> survives a projector. Neither changes what any command prints —
          only which escape codes wrap it.
        </p>
      </Callout>

      <h2>Turning colour off</h2>

      <p>
        effGen honours the <code>NO_COLOR</code> convention: set it to any value and every command
        renders its tables, panels and logo with no colour, whatever theme is selected. Colour is
        also off on its own when stdout is not a terminal — a pipe, a file, a CI log — and when{' '}
        <code>TERM=dumb</code>.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`NO_COLOR=1 effgen models list`}
      />

      <h2>Live animation</h2>

      <p>
        Long commands draw a status line and progress bars on a terminal. Turning the animation off
        leaves plain single-line status text that is safe to pipe or to append to a log.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen run "..." --no-animation
export EFFGEN_NO_ANIM=1`}
        caption={
          <>
            <code>NO_COLOR</code> and a non-interactive stdout also disable it.{' '}
            <code>effgen top --no-animation</code> means something slightly stronger — it prints one
            static snapshot instead of taking over the screen.
          </>
        }
      />

      <h2>The bare landing and the first-run welcome</h2>

      <p>
        Running <code>effgen</code> with no command, on a terminal where both input and output are
        terminals, shows a short landing: the logo, the version, and a set of one-key actions.
        Pressing Enter continues into the interactive setup wizard.
      </p>

      <Terminal
        command="effgen"
        output={`
        ██  ██   ▄████▄
 ▄███▄ ▀██▀▀██▀ ██▀  ▀▀  ▄███▄  ██▄███▄
 ██▄██  ██  ██  ██  ███  ██▄██  ██▀ ▀██
 ██▄▄▄  ██  ██  ▀██▄▄██  ██▄▄▄  ██   ██
  ▀▀▀▀  ▀▀  ▀▀    ▀▀▀▀    ▀▀▀▀  ▀▀   ▀▀

effGen v1.0.1 · agents on small (and cloud) models
Python 3.11.15 · theme: default · docs.effgen.org

What next?
  [Enter]  set up and run an agent (interactive wizard)
  [c]  chat — open an interactive session
  [e]  code — write, run and fix code in this directory
  [q]  quickstart — a 2-minute guided first run
  [d]  doctor — check which providers are ready
  [m]  models — browse the model catalog
  [h]  help — all commands
  [x]  exit

choice ❯`}
        maxLines={24}
        caption="Captured on a real pseudo-terminal — the landing is drawn only when stdin and stdout are both terminals. Piped, redirected, under --quiet or in CI, the command behaves as it always did. The header line names the active theme."
      />

      <p>
        The logo uses a Unicode block banner where the terminal can render one and a plain ASCII
        banner otherwise, and collapses to a one-line wordmark on a narrow terminal. The first time{' '}
        <code>effgen</code> runs on an interactive terminal it also prints a one-time welcome
        pointing at <code>effgen doctor</code>, then records a flag under <code>$EFFGEN_HOME</code>{' '}
        so it never appears again. That welcome is silent under <code>--quiet</code>, in CI and on
        non-interactive output.
      </p>

      <h2>How an answer is presented</h2>

      <p>
        <code>effgen run</code> and <code>effgen chat</code> share one presentation, so a one-shot
        run and a conversational turn read as the same tool.
      </p>

      <ApiTable
        headers={['Part', 'What it does']}
        rows={[
          [
            'The answer',
            <>
              <code>run</code> frames the finished answer in a bordered panel; <code>chat</code>{' '}
              shows it inline under an <code>assistant</code> label. Both render markdown —
              headings, lists, fenced code and tables — through the same renderer, in the selected
              theme.
            </>,
          ],
          [
            'Streaming',
            <>
              With <code>--stream</code>, and in <code>chat</code>, the answer renders as it
              arrives: a brief <code>Thinking…</code> spinner until the first token, then a markdown
              region that updates in place. On a pipe, a redirect or with <code>NO_COLOR</code> it
              falls back to plain token-by-token text with no spinner and no cursor control.
            </>,
          ],
          [
            'The summary',
            <>
              One line at the end — <code>✓ Done in 3.2s · 2 tools · 1,204 tokens · $0.0006</code>.
              A run stopped at its iteration cap is marked partial, and a failure names its reason.{' '}
              <code>chat</code> shows a per-turn footer with a running session total.
            </>,
          ],
          [
            'A run with no answer',
            <>
              A run that hit its iteration cap has no answer to frame. The panel says what stopped
              it and what to do, and whatever the run had reached follows below, labelled as partial
              progress — so retrieved passages are never read as a result.
            </>,
          ],
        ]}
      />

      <p>
        <code>--quiet</code> prints the answer alone, with no header, spinner or summary. Under{' '}
        <code>--json</code> stdout stays one JSON document and all of the above is suppressed. On a
        terminal that cannot encode UTF-8 — <code>PYTHONIOENCODING=ascii</code>, for instance — the
        status glyphs and separators fall back to ASCII stand-ins rather than failing.
      </p>

      <h2>Tips</h2>

      <p>
        effGen prints an occasional one-line tip after a command you watched interactively. Silence
        them for good with:
      </p>

      <CodeBlock language="bash" filename="terminal" code={`export EFFGEN_TIPS=0`} />

      <h2>Every switch, in one table</h2>

      <ApiTable
        headers={['Set this', 'And']}
        rows={[
          [
            <code>NO_COLOR</code>,
            'All colour is off; structure still renders. Also disables animation.',
          ],
          [
            <code>EFFGEN_THEME</code>,
            <>
              Selects a theme —{' '}
              {themes.map((theme, i) => (
                <span key={theme}>
                  {i > 0 && ', '}
                  <code>{theme}</code>
                </span>
              ))}
              . Same values as <code>--theme</code>.
            </>,
          ],
          [<code>EFFGEN_NO_ANIM=1</code>, 'No live spinners or progress bars.'],
          [<code>EFFGEN_TIPS=0</code>, 'No rotating tips.'],
          [
            <code>EFFGEN_HOME</code>,
            <>
              Relocates the per-user state directory, default <code>~/.effgen</code> — which is
              where the first-run flag is recorded.
            </>,
          ],
          [<code>TERM=dumb</code>, 'Treated as a terminal that cannot draw: no colour.'],
          [<code>CI=1</code>, 'No landing, no welcome, no full-screen view.'],
          [
            <code>--no-animation</code>,
            <>
              Per command. On <code>effgen top</code> it also means "one static snapshot and exit".
            </>,
          ],
          [
            <code>-q, --quiet</code>,
            'Per command: the result alone, no chrome. Errors still print.',
          ],
        ]}
        caption={
          <>
            The environment variables are the ones{' '}
            <code>effgen --help</code> and the framework's appearance documentation declare. Where
            two disagree, the more restrictive wins: <code>NO_COLOR</code> beats a theme.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'Escape codes in a captured log',
            'Something is forcing colour onto a non-terminal — usually a pty wrapper, a CI runner that allocates one, or a `script` capture.',
            <>
              <code>NO_COLOR=1</code> in the job's environment. It is honoured even where effGen
              believes it has a terminal.
            </>,
          ],
          [
            <>
              <code>--theme</code> appears to do nothing
            </>,
            <>
              <code>NO_COLOR</code> is set somewhere, or the output is not a terminal. Both drop the
              hue and keep the structure.
            </>,
            <>
              <code>echo $NO_COLOR</code>, and check whether the command is inside a pipe.
            </>,
          ],
          [
            'Boxes render as question marks or garbage',
            'The terminal cannot encode the box-drawing characters.',
            <>
              effGen already falls back to ASCII on a non-UTF-8 stdout; if the locale claims UTF-8
              and the font does not have the glyphs, use <code>--theme monochrome</code> or a font
              with box-drawing coverage.
            </>,
          ],
          [
            'A script hangs on `effgen` with no arguments',
            'A terminal was allocated, so the interactive landing is waiting for a keypress.',
            <>
              Name a command. In a wrapper that may allocate a pty, add <code>--quiet</code> or
              redirect stdin from <code>/dev/null</code>.
            </>,
          ],
          [
            'The progress bar overwrites earlier output in a log',
            'The animation is drawing carriage returns into a file.',
            <>
              <code>--no-animation</code> or <code>EFFGEN_NO_ANIM=1</code>. A plain redirect
              normally disables it already, so this means the process believes it has a terminal.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          The themes, the landing, the first-run welcome and the tips are all new in this release,
          and the shared <code>run</code>/<code>chat</code> answer presentation replaced two
          different ones. The default look is unchanged, so a terminal that says nothing gets what
          it got before.
        </p>
      </Callout>

      <SeeAlso paths={['/cli', '/cli/run', '/cli/top']} />
    </DocPage>
  );
}
