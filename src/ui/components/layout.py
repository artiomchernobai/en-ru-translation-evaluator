import streamlit as st

def fit_viewers() -> None:
    # Observe the actual dock and panel positions rather than assuming a screen size.
    st.html(r"""
    <style>
    [class*="st-key-viewer_body_"] {height: var(--viewer-height, 50dvh) !important; min-height: 160px !important; max-height: var(--viewer-height, 50dvh) !important; flex: 0 0 var(--viewer-height, 50dvh) !important;}
    div:has(> [class*="st-key-viewer_body_"]) {height: var(--viewer-height, 50dvh) !important; flex: 0 0 var(--viewer-height, 50dvh) !important;}
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] label {color: var(--reading-color);}
    .st-key-right_actions {overflow-y: auto; max-height: var(--controls-height, none);}
    [class*="st-key-viewer_body_"] textarea:disabled,
    [data-testid="stBottom"] textarea:disabled {
        color: var(--reading-color) !important;
        -webkit-text-fill-color: var(--reading-color) !important;
        opacity: 1 !important;
    }
    </style>
    <script>
    if (window.__fitViewersCleanup) window.__fitViewersCleanup();
    let scheduled = false;
    const fit = () => {
      scheduled = false;
      const app = document.querySelector('[data-testid="stApp"]');
      if (!app) return;
      const rgb = getComputedStyle(app).backgroundColor.match(/\d+/g);
      const color = rgb && (Number(rgb[0])*.299 + Number(rgb[1])*.587 + Number(rgb[2])*.114) < 128 ? '#ffffff' : '#000000';
      if (document.documentElement.style.getPropertyValue('--reading-color') !== color)
        document.documentElement.style.setProperty('--reading-color', color);
      const dock = document.querySelector('[data-testid="stBottom"]');
      const bottom = Math.min(window.innerHeight, dock ? dock.getBoundingClientRect().top : window.innerHeight);
      const controls = document.querySelector('.st-key-right_actions');
      if (controls) {
        const h = Math.max(200, bottom - controls.getBoundingClientRect().top - 16) + 'px';
        if (controls.style.getPropertyValue('--controls-height') !== h) controls.style.setProperty('--controls-height',h);
      }
      document.querySelectorAll('[class*="st-key-viewer_body_"]').forEach(el => {
        const h = Math.max(160, Math.floor(bottom - el.getBoundingClientRect().top - 52)) + 'px';
        for (const box of [el, el.parentElement]) {
          if (box.style.getPropertyValue('--viewer-height') !== h) box.style.setProperty('--viewer-height', h);
        }
      });
    };
    const schedule = () => { if (!scheduled) {scheduled = true; requestAnimationFrame(fit);} };
    const observer = new ResizeObserver(schedule);
    const dock = document.querySelector('[data-testid="stBottom"]');
    if (dock) observer.observe(dock);
    const theme = new MutationObserver(schedule);
    const app = document.querySelector('[data-testid="stApp"]');
    if (app) theme.observe(app, {attributes:true, attributeFilter:['style','class']});
    window.addEventListener('resize', schedule);
    window.__fitViewersCleanup = () => {observer.disconnect();theme.disconnect();window.removeEventListener('resize',schedule);};
    schedule();
    </script>
    """, unsafe_allow_javascript=True)
