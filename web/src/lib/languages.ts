// Languages a household can choose for the wearer's spoken answers, each named in its own
// language. Keep in step with LANGUAGES in backend/app/languages.py (a test checks).
export const LANGUAGES: { code: string; name: string }[] = [
  { code: "en", name: "English" },
  { code: "es", name: "Español" },
  { code: "zh", name: "中文" },
  { code: "vi", name: "Tiếng Việt" },
  { code: "tl", name: "Tagalog" },
  { code: "fr", name: "Français" },
  { code: "ar", name: "العربية" },
  { code: "ko", name: "한국어" },
  { code: "ru", name: "Русский" },
  { code: "pt", name: "Português" },
  { code: "hi", name: "हिन्दी" },
  { code: "de", name: "Deutsch" },
  { code: "ht", name: "Kreyòl ayisyen" },
];

export const DEFAULT_LANGUAGE = "en";
