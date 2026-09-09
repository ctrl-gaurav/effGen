// Every count this site states, in one place, derived from the installed
// framework by `scripts/gen_site_data.py`. Nothing here is typed by hand.
//
// The documentation site imports the same JSON through its `@data` alias, so
// the two halves of the site cannot disagree about a number.

import raw from "@/data/effgen.json";
import type { SiteData } from "@/data/siteData.types";

export const siteData = raw as unknown as SiteData;

/** `1.0.1` — what the package on PyPI is at the moment this was generated. */
export const version = siteData.version;

/** 66 built-in tools, across 8 categories. */
export const toolCount = siteData.tools.count;

/** 9 ready-to-use agent presets. */
export const presetCount = siteData.presets.count;

/**
 * 10 provider adapters are registered. Nine of them ship a bundled catalog;
 * `openai_compatible` ships none, because it serves whatever the endpoint it is
 * pointed at serves. Both figures are true, so say which one you mean.
 */
export const providerCount = siteData.models.adapter_count;
export const providersWithCatalog = siteData.models.with_catalog_count;

/** 417 catalogued models, across the nine providers that carry a catalog. */
export const modelCount = siteData.models.models;

/** 29 top-level commands and 38 sub-commands. */
export const commandCount = siteData.cli.command_count;
export const subcommandCount = siteData.cli.subcommand_count;

/** 26 slash commands in a coding session; `effgen chat` has its own 13. */
export const codeSlashCommandCount = siteData.code.slash_command_count;
export const chatSlashCommandCount = siteData.code.chat_slash_command_count;

/** 223 names exported from the top-level `effgen` package. */
export const publicNameCount = siteData.public_names;

/** `3.11` through `3.14`. */
export const pythonVersions = siteData.python_versions;

export type { SiteData } from "@/data/siteData.types";
