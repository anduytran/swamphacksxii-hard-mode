import { describe, expect, it } from "vitest";
import { DEFAULT_LANGUAGE, LANGUAGES } from "../lib/languages";
import py from "../../../backend/app/languages.py?raw";

describe("household languages", () => {
  it("has unique codes, English first and the default", () => {
    const codes = LANGUAGES.map((l) => l.code);
    expect(new Set(codes).size).toBe(codes.length);
    expect(codes[0]).toBe(DEFAULT_LANGUAGE);
    expect(DEFAULT_LANGUAGE).toBe("en");
  });

  it("names each language in its own language", () => {
    const name = (code: string) => LANGUAGES.find((l) => l.code === code)?.name;
    expect(name("es")).toBe("Español");
    expect(name("vi")).toBe("Tiếng Việt");
    expect(name("zh")).toBe("中文");
  });

  it("matches the codes the backend accepts", () => {
    const block = py.slice(py.indexOf("LANGUAGES: dict"), py.indexOf("\n}\n"));
    const backend = [...block.matchAll(/^\s+"([a-z-]+)": \(/gm)].map((m) => m[1]);
    expect(backend).toEqual(LANGUAGES.map((l) => l.code));
  });
});
