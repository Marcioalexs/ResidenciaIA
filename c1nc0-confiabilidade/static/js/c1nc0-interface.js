(() => {
  const form = document.getElementById('analysis-form');
  const urlInput = document.getElementById('news-url');
  const clearButton = document.getElementById('clear-url');
  const analyzeButton = document.getElementById('analyze-button');
  const loading = document.getElementById('analysis-loading');

  const updateClearButton = () => {
    if (!clearButton || !urlInput) return;
    clearButton.classList.toggle('visible', Boolean(urlInput.value.trim()));
  };

  if (urlInput) {
    updateClearButton();
    urlInput.addEventListener('input', updateClearButton);
  }

  if (clearButton && urlInput) {
    clearButton.addEventListener('click', () => {
      urlInput.value = '';
      updateClearButton();
      urlInput.focus();
    });
  }

  if (form && urlInput) {
    form.addEventListener('submit', (event) => {
      try {
        const parsed = new URL(urlInput.value.trim());
        if (!['http:', 'https:'].includes(parsed.protocol) || !parsed.hostname.includes('.')) {
          throw new Error('invalid-url');
        }
      } catch {
        event.preventDefault();
        urlInput.setCustomValidity('Insira um endereço completo, incluindo https://');
        urlInput.reportValidity();
        urlInput.addEventListener('input', () => urlInput.setCustomValidity(''), { once: true });
        return;
      }

      if (analyzeButton) {
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<span aria-hidden="true">◌</span> LENDO';
      }
      if (loading) loading.classList.add('visible');
    });
  }

  document.querySelectorAll('[data-format-date]').forEach((element) => {
    const raw = element.getAttribute('data-format-date');
    if (!raw) return;
    const dateOnly = raw.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    const parsed = dateOnly
      ? new Date(Number(dateOnly[1]), Number(dateOnly[2]) - 1, Number(dateOnly[3]))
      : new Date(raw);
    if (Number.isNaN(parsed.getTime())) return;

    const hasTime = /T\d{2}:\d{2}|\d{1,2}:\d{2}/.test(raw);
    const dateText = new Intl.DateTimeFormat('pt-BR', {
      day: '2-digit', month: 'short', year: 'numeric'
    }).format(parsed).replace('.', '');
    const timeText = hasTime
      ? new Intl.DateTimeFormat('pt-BR', { hour: '2-digit', minute: '2-digit' }).format(parsed)
      : '';
    element.textContent = timeText ? `${dateText} · ${timeText}` : dateText;
  });

  const resultStart = document.getElementById('resultado-inicio');
  if (resultStart && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    window.setTimeout(() => resultStart.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
  }

  // Guia do piloto / extensão.
  const pilotOverlay = document.getElementById('pilot-overlay');
  const openPilot = document.getElementById('open-pilot-guide');
  const closePilot = document.getElementById('close-pilot-guide');
  const dontShow = document.getElementById('pilot-dont-show');
  const extensionOk = document.getElementById('extension-ok');
  const extensionNotDetected = document.getElementById('extension-not-detected');
  const showInstall = document.getElementById('show-install-help');
  const checkExtension = document.getElementById('check-extension');
  const pilotKey = 'c1nc0_pilot_guide_hidden_v1';
  let extensionDetected = false;

  const setExtensionState = (installed) => {
    extensionDetected = installed;
    if (extensionOk) extensionOk.style.display = installed ? 'block' : 'none';
    if (extensionNotDetected) extensionNotDetected.style.display = installed ? 'none' : 'block';
    document.querySelectorAll('.extension-install').forEach((el) => {
      el.style.display = installed ? 'none' : '';
    });
    if (showInstall) showInstall.style.display = installed ? 'inline-flex' : 'none';
  };

  const pingExtension = () => window.postMessage({ type: 'C1NC0_EXTENSION_PING' }, '*');
  const showPilot = () => {
    if (!pilotOverlay) return;
    pilotOverlay.classList.add('visible');
    pingExtension();
    window.setTimeout(pingExtension, 300);
    window.setTimeout(pingExtension, 900);
  };
  const hidePilot = () => {
    if (!pilotOverlay) return;
    if (dontShow?.checked) localStorage.setItem(pilotKey, '1');
    pilotOverlay.classList.remove('visible');
  };

  openPilot?.addEventListener('click', showPilot);
  closePilot?.addEventListener('click', hidePilot);
  pilotOverlay?.addEventListener('click', (event) => {
    if (event.target === pilotOverlay) hidePilot();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      hidePilot();
      document.getElementById('feedback-overlay')?.classList.remove('visible');
    }
  });
  showInstall?.addEventListener('click', () => {
    document.querySelectorAll('.extension-install').forEach((el) => { el.style.display = ''; });
    showInstall.style.display = 'none';
  });
  checkExtension?.addEventListener('click', () => {
    extensionDetected = false;
    checkExtension.textContent = 'Verificando...';
    pingExtension();
    window.setTimeout(pingExtension, 400);
    window.setTimeout(() => {
      checkExtension.textContent = extensionDetected ? '✓ Extensão detectada' : 'Verificar extensão';
      setExtensionState(extensionDetected);
    }, 1100);
  });
  window.addEventListener('message', (event) => {
    if (event.source === window && event.data?.type === 'C1NC0_EXTENSION_PONG') {
      setExtensionState(true);
    }
  });

  if (pilotOverlay) {
    pingExtension();
    window.setTimeout(pingExtension, 350);
    window.setTimeout(pingExtension, 1000);
    window.setTimeout(() => { if (!extensionDetected) setExtensionState(false); }, 1500);
    if (pilotOverlay.dataset.autoOpen === '1' && localStorage.getItem(pilotKey) !== '1') showPilot();
  }
})();
