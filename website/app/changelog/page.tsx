import type { Metadata } from "next";
import ChangelogView from "./ChangelogView";
import { siteData, version } from "@/components/siteData";
import { pageMetadata } from "@/components/seo";
import { RELEASE_DATE_1_3_0 } from "./changelogData";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description — the same shape the
// example detail pages use.
export const metadata: Metadata = pageMetadata({
  path: "/changelog",
  card: "changelog",
  title: "Changelog",
  description:
    `effGen ${version}, released ${RELEASE_DATE_1_3_0}: the ten changes existing code sees, ` +
    `${siteData.public_names} public names, and every earlier release.`,
});

export default function Page() {
  return <ChangelogView />;
}
