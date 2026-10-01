(() => {
  const MIN_SCORE = 5;
  const DISMISS_KEY = `c1nc0-dismissed:${location.href}`;

  // Permite que a aplicação C1NC0 detecte localmente que a extensão está ativa.
  // Nenhum dado de navegação é enviado nessa verificação.
  window.addEventListener("message", (event) => {
    if (event.source !== window || event.data?.type !== "C1NC0_EXTENSION_PING") return;
    window.postMessage({ type: "C1NC0_EXTENSION_PONG", version: chrome.runtime.getManifest().version }, "*");
  });

  function jsonLdTypes() {
    const types = [];
    document.querySelectorAll('script[type="application/ld+json"]').forEach((node) => {
      try {
        const parsed = JSON.parse(node.textContent || "null");
        const queue = Array.isArray(parsed) ? [...parsed] : [parsed];
        while (queue.length) {
          const item = queue.shift();
          if (!item || typeof item !== "object") continue;
          if (Array.isArray(item["@graph"])) queue.push(...item["@graph"]);
          const value = item["@type"];
          if (Array.isArray(value)) types.push(...value);
          else if (value) types.push(value);
        }
      } catch (_) {
        // JSON-LD inválido não impede os demais sinais de serem avaliados.
      }
    });
    return types.map((value) => String(value).toLowerCase());
  }

  function metaContent(selector) {
    return document.querySelector(selector)?.getAttribute("content")?.trim() || "";
  }

  function detectJournalisticPage() {
    let score = 0;
    const signals = [];
    const types = jsonLdTypes();

    if (types.some((type) => ["newsarticle", "article", "reportagenewsarticle", "analysisnewsarticle"].includes(type))) {
      score += 4;
      signals.push("dados estruturados Article/NewsArticle");
    }

    if (metaContent('meta[property="og:type"]').toLowerCase() === "article") {
      score += 2;
      signals.push("og:type=article");
    }

    if (document.querySelector("article")) {
      score += 2;
      signals.push("elemento <article>");
    }

    if (metaContent('meta[property="article:published_time"]') || document.querySelector("time[datetime]")) {
      score += 2;
      signals.push("data de publicação");
    }

    if (
      metaContent('meta[name="author"]') ||
      metaContent('meta[property="article:author"]') ||
      document.querySelector('[rel="author"], [itemprop="author"]')
    ) {
      score += 1;
      signals.push("autoria");
    }

    if (document.querySelector("h1")) {
      score += 1;
      signals.push("título H1");
    }

    const article = document.querySelector("article");
    const textLength = (article?.innerText || document.body?.innerText || "").trim().length;
    if (textLength >= 800) {
      score += 1;
      signals.push("texto extenso");
    }

    return { score, signals, likely: score >= MIN_SCORE };
  }

  function showSuggestion(detection) {
    if (sessionStorage.getItem(DISMISS_KEY) === "1") return;
    if (document.getElementById("c1nc0-suggestion")) return;

    const box = document.createElement("aside");
    box.id = "c1nc0-suggestion";
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-label", "Sugestão do C1NC0");

    const title = document.createElement("strong");
    title.textContent = "C1NC0 — Pensamento Crítico";

    const text = document.createElement("p");
    text.textContent = "Esta página apresenta características de conteúdo jornalístico. Quer usar o C1NC0 para orientar sua análise?";

    const detail = document.createElement("p");
    detail.className = "c1nc0-detail";
    detail.textContent = `Detecção local: ${detection.score} pontos. A URL só será enviada ao C1NC0 se você escolher analisar.`;

    const actions = document.createElement("div");
    actions.className = "c1nc0-actions";

    const analyze = document.createElement("button");
    analyze.type = "button";
    analyze.className = "c1nc0-primary";
    analyze.textContent = "Analisar com C1NC0";
    analyze.addEventListener("click", () => {
      chrome.runtime.sendMessage({ type: "C1NC0_OPEN_ANALYSIS", url: location.href });
      box.remove();
    });

    const dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.className = "c1nc0-secondary";
    dismiss.textContent = "Agora não";
    dismiss.addEventListener("click", () => {
      sessionStorage.setItem(DISMISS_KEY, "1");
      box.remove();
    });

    actions.append(analyze, dismiss);
    box.append(title, text, detail, actions);
    document.documentElement.appendChild(box);
  }

  const detection = detectJournalisticPage();
  if (detection.likely) showSuggestion(detection);
})();
