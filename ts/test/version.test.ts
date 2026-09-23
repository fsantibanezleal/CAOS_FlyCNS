import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { VERSION } from "../src/index.js";

test("the exported version equals the npm manifest and the VERSION file", () => {
  const root = new URL("../../", import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("package.json", root), "utf-8")) as { version: string };
  const display = readFileSync(new URL("VERSION", root), "utf-8").trim();
  const semver = display.split(".").map((part) => String(Number(part))).join(".");
  assert.equal(VERSION, manifest.version);
  assert.equal(VERSION, semver);
});
