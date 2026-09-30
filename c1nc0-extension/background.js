const C1NC0_BASE_URL = "https://residencia-ia-2r6h.vercel.app/";

function buildAnalysisUrl(pageUrl) {
  const target = new URL(C1NC0_BASE_URL);
  target.searchParams.set("url", pageUrl);
  target.searchParams.set("analisar", "1");
  target.searchParams.set("origem", "extensao");
  return target.toString();
}

chrome.runtime.onMessage.addListener((message) => {
  if (message?.type !== "C1NC0_OPEN_ANALYSIS" || !message.url) return;

  try {
    const pageUrl = new URL(message.url);
    if (!["http:", "https:"].includes(pageUrl.protocol)) return;
    chrome.tabs.create({ url: buildAnalysisUrl(pageUrl.toString()) });
  } catch (error) {
    console.warn("C1NC0: URL inválida.", error);
  }
});
