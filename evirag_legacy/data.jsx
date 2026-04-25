/* global React */
// EVIRAG sample data — domain-specific scientific controversy

const SAMPLE_QUERIES = {
  "Frontier": [
    { q: "Is overparameterization necessary for generalization in deep neural networks?", meta: "ML · 47 claims · highly disputed" },
    { q: "Does scaling alone produce emergent reasoning, or is it a measurement artifact?", meta: "AI · 31 claims · contested" },
    { q: "Are mechanistic interpretability methods faithful to model internals?", meta: "Interp · 24 claims · disputed" }
  ],
  "Health": [
    { q: "Does intermittent fasting improve metabolic outcomes beyond caloric restriction?", meta: "Nutrition · 38 claims · highly disputed" },
    { q: "Is moderate alcohol consumption protective against cardiovascular disease?", meta: "Cardiology · 52 claims · contested" },
    { q: "Do statins meaningfully reduce all-cause mortality in primary prevention?", meta: "Pharma · 41 claims · disputed" }
  ],
  "Climate": [
    { q: "What is the equilibrium climate sensitivity range to a doubling of CO₂?", meta: "Climate · 29 claims · narrowing" },
    { q: "Are current carbon-removal cost projections empirically supported?", meta: "CDR · 22 claims · highly disputed" }
  ],
  "Economics": [
    { q: "Do minimum wage increases reduce employment in low-wage sectors?", meta: "Labor · 64 claims · contested" },
    { q: "Is the natural rate of unemployment a useful policy anchor in 2026?", meta: "Macro · 18 claims · disputed" }
  ],
  "Method": [
    { q: "Are pre-registered replications a sufficient remedy for the replication crisis?", meta: "Meta-sci · 27 claims · disputed" },
    { q: "Does p < 0.005 meaningfully address false discovery without raising power problems?", meta: "Stats · 19 claims · contested" }
  ]
};

const RECENT_THREADS = [
  { id: "t1", title: "Equilibrium climate sensitivity to 2× CO₂", time: "2h", level: "med" },
  { id: "t2", title: "Statins for primary prevention — mortality endpoint", time: "yest", level: "high" },
  { id: "t3", title: "Mechanistic interpretability faithfulness audits", time: "3d", level: "med" },
  { id: "t4", title: "Replication rates in social priming literature", time: "5d", level: "low" }
];

// A canned result for the demo query
const DEMO_RESULT = {
  query: "Is overparameterization necessary for generalization in deep neural networks?",
  intent: { dispute: "highly_disputed", domain: "machine learning", expectedDisagreement: "high" },
  metrics: {
    confidence: 0.62,
    confidenceLabel: "MEDIUM",
    disagreementDensity: 0.183,
    conflictRatio: 0.65,
    claimEntropy: 1.42,
    visualMismatch: 0.30,
    claims: 47,
    sources: 34,
    figuresAligned: 12,
    controversyClass: "stable"
  },
  views: [
    {
      kind: "dom",
      label: "Dominant view",
      sources: 14,
      confidence: 0.71,
      summary: "Overparameterization confers conditional benefits via implicit regularization and the double-descent regime, particularly when training dynamics are SGD-driven and data is non-trivially structured.",
      claims: [
        { n: "C-04", text: "Implicit regularization in gradient descent biases solutions toward low-norm minima even at extreme width.", src: "Belkin '19", cites: ["S1", "S2"] },
        { n: "C-09", text: "Test error exhibits a second descent past the interpolation threshold across vision and tabular benchmarks.", src: "Nakkiran '20", cites: ["S3"] },
        { n: "C-12", text: "Effective parameter counts under NTK linearization are far smaller than nominal capacity.", src: "Jacot '18", cites: ["S4", "S5"] }
      ]
    },
    {
      kind: "alt",
      label: "Alternative view",
      sources: 9,
      confidence: 0.54,
      summary: "Generalization is bottlenecked by data quality, augmentation, and architectural inductive bias — width is a coarse proxy for capacity that obscures the actual mechanism.",
      claims: [
        { n: "C-21", text: "Smaller models with strong augmentation match overparameterized baselines on ImageNet at matched FLOPs.", src: "Touvron '22", cites: ["S6"] },
        { n: "C-27", text: "Inductive bias from convolution accounts for most of the generalization gap attributed to scale.", src: "d'Ascoli '21", cites: ["S7"] }
      ]
    },
    {
      kind: "min",
      label: "Minority / contrarian",
      sources: 4,
      confidence: 0.41,
      summary: "Classical bias–variance reasoning still applies once measurement is corrected for label noise and benchmark leakage.",
      claims: [
        { n: "C-33", text: "Once benchmark contamination is filtered, overparameterized models show no advantage on held-out distributions.", src: "Recht '19", cites: ["S8"] }
      ]
    }
  ],
  agents: [
    { glyph: "P", name: "Precision", role: "Strict semantic · high-confidence support", retrieved: 5, kept: 5, model: "llama3:latest" },
    { glyph: "R", name: "Recall", role: "Expansive semantic · broad coverage", retrieved: 15, kept: 12, model: "qwen2.5:4b" },
    { glyph: "S", name: "Skeptic", role: "Adversarial · contradiction-seeking", retrieved: 8, kept: 7, model: "llama3:latest" },
    { glyph: "C", name: "Counterfactual", role: "Alternative explanations", retrieved: 6, kept: 5, model: "deepseek-r1:7b" }
  ],
  sources: [
    { n: 1, title: "Reconciling modern machine-learning practice and the classical bias–variance trade-off", venue: "PNAS", year: 2019, stance: "support" },
    { n: 2, title: "Deep double descent: where bigger models and more data hurt", venue: "ICLR", year: 2020, stance: "support" },
    { n: 3, title: "Neural tangent kernel: convergence and generalization in neural networks", venue: "NeurIPS", year: 2018, stance: "support" },
    { n: 4, title: "ResNet strikes back: an improved training procedure in timm", venue: "arXiv", year: 2022, stance: "contra" },
    { n: 5, title: "ConViT: improving vision transformers with soft convolutional inductive biases", venue: "ICML", year: 2021, stance: "contra" },
    { n: 6, title: "Do ImageNet classifiers generalize to ImageNet?", venue: "ICML", year: 2019, stance: "contra" },
    { n: 7, title: "On the role of width in implicit regularization of deep linear networks", venue: "JMLR", year: 2023, stance: "neutral" }
  ],
  graph: {
    nodes: [
      { id: "n1", x: 0.18, y: 0.30, kind: "dom", label: "C-04" },
      { id: "n2", x: 0.32, y: 0.18, kind: "dom", label: "C-09" },
      { id: "n3", x: 0.46, y: 0.34, kind: "dom", label: "C-12" },
      { id: "n4", x: 0.62, y: 0.20, kind: "alt", label: "C-21" },
      { id: "n5", x: 0.74, y: 0.40, kind: "alt", label: "C-27" },
      { id: "n6", x: 0.55, y: 0.62, kind: "min", label: "C-33" },
      { id: "n7", x: 0.32, y: 0.70, kind: "dom", label: "C-17" },
      { id: "n8", x: 0.84, y: 0.68, kind: "alt", label: "C-29" }
    ],
    edges: [
      { a: "n1", b: "n2", kind: "support" },
      { a: "n1", b: "n3", kind: "support" },
      { a: "n2", b: "n3", kind: "support" },
      { a: "n3", b: "n4", kind: "contra" },
      { a: "n2", b: "n5", kind: "contra" },
      { a: "n4", b: "n5", kind: "support" },
      { a: "n6", b: "n1", kind: "contra" },
      { a: "n6", b: "n2", kind: "contra" },
      { a: "n7", b: "n1", kind: "support" },
      { a: "n7", b: "n6", kind: "neutral" },
      { a: "n5", b: "n8", kind: "support" },
      { a: "n8", b: "n3", kind: "neutral" }
    ]
  }
};

window.SAMPLE_QUERIES = SAMPLE_QUERIES;
window.RECENT_THREADS = RECENT_THREADS;
window.DEMO_RESULT = DEMO_RESULT;
