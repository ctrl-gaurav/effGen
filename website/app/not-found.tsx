import RouteLink from "@/components/ui/RouteLink";
import type { Metadata } from "next";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Container from "@/components/Container";
import { pageMetadata } from "@/components/seo";

export const metadata: Metadata = pageMetadata({
  path: "/404",
  title: "Page not found",
  description:
    "That page does not exist. The most useful places to go instead are the documentation, " +
    "the quick start, the examples and the changelog.",
  // A 404 that gets indexed competes with the page the reader wanted.
  noindex: true,
});

// Where someone who hit a dead URL most likely meant to go. A 404 that only
// says "not found" makes the reader go back to the navigation and start again;
// these are the four destinations that cover almost every wrong URL on this
// site.
const destinations = [
  {
    href: "/docs",
    title: "Documentation",
    blurb: "Every behaviour, every flag, every parameter, with runnable code.",
  },
  {
    href: "/docs/quickstart",
    title: "Quick start",
    blurb: "Install it and get an agent answering in a couple of minutes.",
  },
  {
    href: "/examples",
    title: "Examples",
    blurb: "Complete programs with the output they actually produce.",
  },
  {
    href: "/changelog",
    title: "Changelog",
    blurb: "What changed in 1.3.0, and in every release before it.",
  },
];

export default function NotFound() {
  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />
      <main id="main">
        <Container className="pt-40 pb-28">
          <div className="max-w-2xl mx-auto text-center">
            <p className="font-mono text-sm text-green-700 dark:text-green-400 mb-4">
              404
            </p>
            <h1 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
              That page <span className="gradient-text">does not exist</span>
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-400">
              The link may be out of date, or the address may have a typo in it.
              Nothing is broken on your side.
            </p>
          </div>

          <h2 className="sr-only">Where to go next</h2>
          <div className="mt-14 grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-3xl mx-auto">
            {destinations.map((destination) => (
              <RouteLink
                key={destination.href}
                to={destination.href}
                className="block p-6 rounded-2xl border border-gray-200 dark:border-green-500/20 bg-gray-50 dark:bg-[#04140c] hover:border-green-500/50 transition-colors"
              >
                <span className="block font-bold text-gray-900 dark:text-white mb-1.5">
                  {destination.title}
                </span>
                <span className="block text-sm text-gray-600 dark:text-gray-400">
                  {destination.blurb}
                </span>
              </RouteLink>
            ))}
          </div>

          <p className="mt-12 text-center text-sm text-gray-600 dark:text-gray-400">
            If a link on this site brought you here,{" "}
            <a
              href="https://github.com/ctrl-gaurav/effGen/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="text-green-700 dark:text-green-400 underline underline-offset-2"
            >
              tell us where it was
            </a>
            .
          </p>
        </Container>
      </main>
      <Footer />
    </div>
  );
}
