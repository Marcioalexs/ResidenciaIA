const button = document.getElementById("analisar");
const status = document.getElementById("status");

button.addEventListener("click", async () => {
  status.textContent = "";
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab?.url || !/^https?:\/\//i.test(tab.url)) {
    status.textContent = "Abra uma página HTTP/HTTPS para analisá-la.";
    return;
  }

  chrome.runtime.sendMessage({ type: "C1NC0_OPEN_ANALYSIS", url: tab.url });
  window.close();
});
