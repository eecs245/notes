// Normalize solution boxes at build time, preserving their content and dropdown state.
const text = (node) => typeof node?.value === "string"
  ? node.value
  : (node?.children ?? []).map(text).join("");

const normalizeSolutions = (node) => {
  if (node?.type === "admonition") {
    const title = node.children?.find((child) => child.type === "admonitionTitle");
    if (/^solutions?\b/i.test(text(title).trim())) {
      node.kind = "tip";
      node.class = [...new Set(`${node.class ?? ""} notes-solution`.trim().split(/\s+/))].join(" ");
    }
  }
  for (const child of node?.children ?? []) normalizeSolutions(child);
};

export default {
  name: "solution-admonitions",
  transforms: [{
    name: "solution-admonitions",
    stage: "document",
    plugin: () => normalizeSolutions,
  }],
};
