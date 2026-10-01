import { ExternalLink, PenLine } from 'lucide-react';
import DocPage, { Lede } from '../components/DocPage';
import { FRAMEWORK_DOCS, pageFor } from '../nav';
import './StubPage.css';

/**
 * What a route in the navigation shows before its page is written.
 *
 * The route exists from the moment it is listed, so the navigation is complete
 * and every link in it goes somewhere. What the reader gets here is the page's
 * title, the sentence saying what it will cover, a plain statement that it is
 * not written yet, and — because that is more use than an apology — a link to
 * the framework's own documentation for the same topic, which is where the page
 * will be written from.
 *
 * It is deliberately loud. A half-finished page that looks finished is worse
 * than one that says what it is.
 */
export default function StubPage({ path }: { path: string }) {
  const page = pageFor(path);
  if (!page) return null;

  return (
    <DocPage subtitle={<Lede text={page.lede} />} toc={false}>
      <div className="stub-notice" role="note">
        <div className="stub-notice-head">
          <PenLine size={18} aria-hidden="true" />
          <span>This page has not been written yet</span>
        </div>
        <p>
          It is part of the documentation for effGen 1.3.0 and is being written. The
          navigation carries it already so that nothing links into a gap.
        </p>
        {page.source && (
          <p>
            In the meantime, the framework's own documentation covers this topic:{' '}
            <a
              href={`${FRAMEWORK_DOCS}/${page.source}`}
              target="_blank"
              rel="noopener noreferrer"
              className="stub-source"
            >
              <code>{page.source}</code>
              <ExternalLink size={13} aria-hidden="true" />
              <span className="sr-only"> (opens on GitHub)</span>
            </a>
          </p>
        )}
      </div>
    </DocPage>
  );
}
