import js from "@eslint/js";

export default [
  {
    ignores: ["node_modules/**", ".venv/**", "docs/**", "prompts/**"],
  },
  js.configs.recommended,
];
