const $ = (id) => document.getElementById(id);

const PLAY_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2.5" y="4.5" width="19" height="15" rx="2.5"/><path d="M2.5 9h19"/><path d="M7 4.5V9M17 4.5V9"/><path d="M10.5 12.6v3.4l3-1.7z" fill="currentColor"/></svg>';

async function api(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

const fetchState = async () => (await fetch('/api/state')).json();

// --- global run lock: only one thing may drive the single simulator at a
// time, so a run on either tab disables every other run control on the page.
// ponytail: one flag + a refresh callback per hero beats per-button bookkeeping.
let busy = false;
const refreshers = [];
function setBusy(on) {
  busy = on;
  for (const refresh of refreshers) refresh();
  $('randomize-btn').disabled = on;
  $('arrange-btn').disabled = on || !$('generalize-seed').dataset.seed;
  $('vla-run').disabled = on;
}

async function stopRun() {
  try { await api('/api/task', { action: 'cancel' }); } catch (err) { /* already idle */ }
}

const previewMode = $('preview-mode');
fetchState().then((state) => { previewMode.value = state.preview_mode || 'hd'; }).catch(() => {});
previewMode.addEventListener('change', async () => {
  previewMode.disabled = true;
  try {
    await api('/api/control', { preview_mode: previewMode.value });
  } catch (err) {
    previewMode.value = 'hd';
  } finally {
    previewMode.disabled = false;
  }
});

// --- tabs ---
let vlaStarted = false;
for (const tab of document.querySelectorAll('.tab')) {
  tab.addEventListener('click', () => {
    for (const other of document.querySelectorAll('.tab')) {
      other.classList.toggle('selected', other === tab);
      other.setAttribute('aria-selected', String(other === tab));
    }
    $('panel-expert').hidden = tab.dataset.tab !== 'expert';
    $('panel-vla').hidden = tab.dataset.tab !== 'vla';
    if (tab.dataset.tab === 'vla' && !vlaStarted && seedEntries.length) {
      vlaStarted = true;
      // The Hub commit history for pjmorales04/duet-micro-v1 confirms the
      // training upload was all 10 seeds / 50 episodes, so the VLA tab
      // legitimately offers the same seed list as the Expert tab.
      vlaHero.init(seedEntries);
    }
  });
}

// --- one modal for both recorded videos and zoomed target stills ---
function openVideo(entry) {
  $('modal-title').textContent = `Seed ${entry.seed} — recorded HD run`;
  $('modal-image').hidden = true;
  $('modal-player').hidden = false;
  $('modal-player').src = `/media/expert/${entry.video}`;
  $('modal').hidden = false;
  $('modal-player').play().catch(() => {});
}
function openImage(src, alt) {
  $('modal-title').textContent = alt || 'Expected final table';
  $('modal-player').pause();
  $('modal-player').hidden = true;
  $('modal-image').hidden = false;
  $('modal-image').src = src;
  $('modal-image').alt = alt || '';
  $('modal').hidden = false;
}
function closeModal() {
  $('modal').hidden = true;
  $('modal-player').pause();
  $('modal-player').src = '';
}
$('modal-close').addEventListener('click', closeModal);
$('modal').addEventListener('click', (event) => { if (event.target === $('modal')) closeModal(); });
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeModal(); });
for (const img of document.querySelectorAll('img[data-zoom]')) {
  img.addEventListener('click', () => openImage(img.src, img.alt));
}

// --- live viewer used by the generalization section ---
let statusTimer = null;
function showLiveViewer(title) {
  $('live-viewer-title').textContent = title;
  $('live-viewer').hidden = false;
  $('live-frame').src = '/stream?' + Date.now();
  $('live-viewer').scrollIntoView({ behavior: 'smooth', block: 'center' });
  clearInterval(statusTimer);
  statusTimer = setInterval(async () => {
    try {
      const task = (await fetchState()).task || {};
      $('live-status').textContent = task.active
        ? `${task.stage_label || task.stage || 'running'} — ${task.message || ''}`
        : (task.message || 'Idle.');
    } catch (err) {
      $('live-status').textContent = 'Lost connection to the simulator.';
    }
  }, 700);
}
$('live-viewer-close').addEventListener('click', () => {
  $('live-viewer').hidden = true;
  clearInterval(statusTimer);
});

// --- shared hero: seed list + live feed + per-arm instruction buttons ---
// Used by both the expert tab (scripted teacher, physics-verified) and the
// VLA tab (real checkpoint, position-verified).
function createHero({ prefix, subtitle, start, verify, duet }) {
  const el = (suffix) => $(`${prefix}-${suffix}`);
  const seedlist = el('seedlist');
  const frame = el('frame');
  const progress = el('progress');
  const status = el('status');
  const targetRef = el('target-ref');
  const loading = el('loading');
  const stopBtn = el('stop');
  const lists = { right: el('right-list'), left: el('left-list'), duet: el('duet-list') };

  let steps = [];       // { skill, instruction, button, done }
  let unlocked = 0;     // index of the next runnable step
  let running = false;

  function refresh() {
    steps.forEach((step, i) => {
      step.button.disabled = busy || step.done || i !== unlocked;
      step.button.classList.toggle('done', step.done);
    });
    stopBtn.disabled = !running;
  }
  refreshers.push(refresh);

  const setLoading = (on) => { loading.hidden = !on; };
  el('target-btn').addEventListener('click', () => {
    targetRef.hidden = !targetRef.hidden;
    el('target-btn').textContent = targetRef.hidden ? 'Show expected final table' : 'Hide expected final table';
  });
  stopBtn.addEventListener('click', stopRun);

  async function waitForIdle() {
    while (true) {
      const state = await fetchState();
      const task = state.task || {};
      if (!task.active) { progress.style.width = '0%'; return state; }
      progress.style.width = `${Math.round((task.progress || 0) * 100)}%`;
      status.textContent = task.message || 'Running…';
      await new Promise((r) => setTimeout(r, 600));
    }
  }

  function addStep({ skill, instruction, side }) {
    const button = document.createElement('button');
    button.textContent = instruction;
    lists[side].appendChild(button);
    const step = { skill, instruction, button, done: false };
    steps.push(step);
    const index = steps.length - 1;
    button.addEventListener('click', async () => {
      running = true;
      setBusy(true);
      status.textContent = `Running: "${instruction}"`;
      setLoading(true);
      try {
        await start(skill, instruction);
        setLoading(false);
        const finalState = await waitForIdle();
        if (verify(finalState, skill)) {
          step.done = true;
          button.textContent = `✓ ${instruction}`;
          unlocked = index + 1;
          status.textContent = unlocked < steps.length ? `Done. ${subtitle}` : 'All steps complete — the table is set.';
        } else {
          status.textContent = 'That attempt did not finish the job — run it again.';
        }
      } catch (err) {
        status.textContent = 'Could not run that step: ' + err.message;
      } finally {
        setLoading(false);
        running = false;
        setBusy(false);
      }
    });
  }

  async function selectSeed(entry) {
    if (busy) return;
    for (const card of seedlist.querySelectorAll('.seed-pick')) {
      card.classList.toggle('selected', card.dataset.seed === entry.seed);
    }
    steps = [];
    unlocked = 0;
    for (const list of Object.values(lists)) list.innerHTML = '';
    status.textContent = 'Setting up the messy table…';
    progress.style.width = '0%';
    // Show the goal before the mess: judges see what "done" looks like
    // (top view + side angle) every time they pick a seed, not just once.
    targetRef.hidden = false;
    el('target-btn').textContent = 'Hide expected final table';
    setLoading(true);
    let state;
    try {
      state = await api('/api/reset', { scenario: 'dinner', seed: Number(entry.seed), dinner_preset: 'task' });
      status.textContent = `Ready. ${subtitle}`;
    } catch (err) {
      status.textContent = 'Could not set up the table: ' + err.message;
      setLoading(false);
      return;
    }
    setLoading(false);
    frame.src = '/stream?' + Date.now();

    // ponytail: skills were collected bottle->plate->mug->fork->spoon, and the
    // fork/spoon approach paths run right past where the plate lands, so an
    // out-of-order run genuinely fails path-clearance checks. Unlock one at a
    // time, split visually by whichever arm's side the object spawned on.
    const bySkill = Object.fromEntries((state.objects || []).map((o) => [o.id, o]));
    for (const [skill, instruction] of Object.entries(entry.instructions || {})) {
      const side = (bySkill[skill]?.initial_position_m?.[0] ?? 0) >= 0 ? 'right' : 'left';
      addStep({ skill, instruction, side });
    }
    if (duet) addStep({ ...duet, side: 'duet' });
    refresh();
  }

  function init(entries) {
    seedlist.innerHTML = '';
    for (const entry of entries) {
      const card = document.createElement('div');
      card.className = 'seed-pick';
      card.dataset.seed = entry.seed;
      card.innerHTML = `
        <button class="seed-name"><strong>Seed ${entry.seed}</strong><span>${entry.changes || ''}</span></button>
        <button class="video-icon" title="Watch the recorded HD run" aria-label="Watch recorded run for seed ${entry.seed}">${PLAY_ICON}</button>`;
      card.querySelector('.seed-name').addEventListener('click', () => selectSeed(entry));
      card.querySelector('.video-icon').addEventListener('click', (event) => {
        event.stopPropagation();
        openVideo(entry);
      });
      seedlist.appendChild(card);
    }
    if (entries[0]) selectSeed(entries[0]);
  }

  return { init };
}

const expertHero = createHero({
  prefix: 'hero',
  subtitle: 'Pick the next instruction to place an item.',
  start: (skill, instruction) => (skill === 'pour'
    ? api('/api/command', { text: 'pour water' })
    : api('/api/task', { action: 'start', kind: 'dinner_place', object_id: skill, arm: 'auto' })),
  verify: (state) => state.task?.status === 'succeeded',
  duet: { skill: 'pour', instruction: 'Pour water from the bottle into the mug.' },
});

// ponytail: the VLA task always reports "succeeded" once its fixed duration
// elapses (see vla_task.py), unlike the expert's dinner_place which has a real
// physical monitor. So gate progression on the object's actual final position.
const vlaHero = createHero({
  prefix: 'vla',
  subtitle: 'Pick an instruction to run the live checkpoint (CPU inference, give it a moment).',
  start: (skill, instruction) => api('/api/command', { text: instruction, mode: 'learned_vla' }),
  verify: (state, skill) => {
    const obj = (state.objects || []).find((o) => o.id === skill);
    const target = (state.targets || []).find((t) => t.object_id === skill);
    if (!obj || !target) return false;
    const dx = obj.position_m[0] - target.position_m[0];
    const dy = obj.position_m[1] - target.position_m[1];
    return obj.above_table && Math.hypot(dx, dy) <= target.radius_m * 1.5;
  },
});

// --- seed manifest: drives both heroes' seed lists, the video modal and the log strip ---
let seedEntries = [];

async function loadGallery() {
  try {
    seedEntries = await (await fetch('/media/expert/manifest.json')).json();
  } catch (err) {
    $('hero-seedlist').innerHTML = '<p class="loading-note">Could not load the expert gallery manifest.</p>';
    return;
  }
  expertHero.init(seedEntries);
  const strip = $('trial-thumbs');
  for (const entry of seedEntries) {
    const button = document.createElement('button');
    button.innerHTML = `${PLAY_ICON}<span>Seed ${entry.seed}</span>`;
    button.addEventListener('click', () => openVideo(entry));
    strip.appendChild(button);
  }
}
loadGallery();

// --- generalization: a random seed the gallery never showed ---
$('randomize-btn').addEventListener('click', async () => {
  const seed = Math.floor(Math.random() * 2_000_000_000) + 1;
  setBusy(true);
  $('generalize-seed').textContent = 'Randomizing…';
  try {
    await api('/api/reset', { scenario: 'dinner', seed, dinner_preset: 'task' });
    $('generalize-seed').textContent = `Seed ${seed} (not in the gallery above)`;
    $('generalize-seed').dataset.seed = seed;
    showLiveViewer(`Live — random seed ${seed}`);
  } catch (err) {
    $('generalize-seed').textContent = 'Could not randomize the table: ' + err.message;
  } finally {
    setBusy(false);
  }
});

$('arrange-btn').addEventListener('click', async () => {
  setBusy(true);
  const bar = $('generalize-progress');
  try {
    await api('/api/task', { action: 'start', kind: 'set_table' });
    showLiveViewer(`Live — seed ${$('generalize-seed').dataset.seed}`);
    while (true) {
      const task = (await fetchState()).task || {};
      if (!task.active) { bar.style.width = '0%'; break; }
      bar.style.width = `${Math.round((task.progress || 0) * 100)}%`;
      await new Promise((r) => setTimeout(r, 600));
    }
  } catch (err) {
    $('live-status').textContent = 'Could not start the arrangement task: ' + err.message;
  } finally {
    setBusy(false);
  }
});

// --- live VLA: free-text fallback ---
$('vla-run').addEventListener('click', async () => {
  const text = $('vla-instruction').value.trim();
  if (!text) return;
  setBusy(true);
  $('vla-status').textContent = 'Running inference — the first call can take a while on CPU…';
  try {
    await api('/api/command', { text, mode: 'learned_vla' });
    showLiveViewer('Live SmolVLA');
  } catch (err) {
    $('vla-status').textContent = 'Could not start: ' + err.message;
  } finally {
    setBusy(false);
  }
});
