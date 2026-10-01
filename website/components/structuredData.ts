import { siteData } from "./siteData";
import { SITE_NAME, SITE_URL } from "./seo";

// What the project is, in the vocabulary schema.org gives search engines.
//
// Two things are described: the site itself, and the package it documents. Every
// field is read from `data/effgen.json` or from the framework's own package
// metadata, so this cannot go on describing a release the rest of the site has
// moved off.
//
// It is emitted once, in the root layout, and applies to every route.

/** The date on the 1.3.0 entry in the framework's changelog, in ISO form. */
const RELEASE_DATE_ISO = "2026-10-01";

const REPOSITORY = "https://github.com/ctrl-gaurav/effGen";
const PACKAGE = "https://pypi.org/project/effgen/";

const website = {
  "@type": "WebSite",
  "@id": `${SITE_URL}/#website`,
  name: `${SITE_NAME} — agents built for small language models`,
  url: `${SITE_URL}/`,
  inLanguage: "en",
};

const project = {
  "@type": "SoftwareApplication",
  "@id": `${SITE_URL}/#software`,
  name: SITE_NAME,
  alternateName: "effgen",
  applicationCategory: "DeveloperApplication",
  softwareVersion: siteData.version,
  datePublished: RELEASE_DATE_ISO,
  operatingSystem: "Linux, macOS, Windows",
  description:
    `A framework for building agents on small language models. ` +
    `${siteData.tools.count} built-in tools, ${siteData.presets.count} presets, ` +
    `${siteData.models.adapter_count} provider adapters and any OpenAI-compatible server, ` +
    `retrieval, memory, guardrails, evaluation, a terminal coding agent and an ` +
    `OpenAI-compatible API server.`,
  url: `${SITE_URL}/`,
  downloadUrl: PACKAGE,
  codeRepository: REPOSITORY,
  license: "https://www.apache.org/licenses/LICENSE-2.0",
  programmingLanguage: "Python",
  runtimePlatform: `Python ${siteData.python_versions[0]}–${siteData.python_versions[siteData.python_versions.length - 1]}`,
  softwareHelp: { "@type": "CreativeWork", url: `${SITE_URL}/docs/` },
  // The package is free and open source; a crawler reads that from an offer of
  // zero rather than from the absence of one.
  offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
};

export const structuredData = {
  "@context": "https://schema.org",
  "@graph": [website, project],
};
