const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

// --- tabs ---
for (const tab of document.querySelectorAll('.tab')) {
  tab.addEventListener('click', () => {
    for (const other of document.querySelectorAll('.tab')) {
      other.classList.toggle('selected', other === tab);
      other.setAttribute('aria-selected', String(other === tab));
    }
    $('panel-expert').hidden = tab.dataset.tab !== 'expert';
    $('panel-vla').hidden = tab.dataset.tab !== 'vla';
  });
}

// --- live viewer shared by every "try live" action ---
let statusTimer = null;

function showLiveViewer(title) {
  $('live-viewer-title').textContent = title;
  $('live-viewer').hidden = false;
  $('live-frame').src = '/stream?' + Date.now();
  $('live-viewer').scrollIntoView({ behavior: 'smooth', block: 'center' });
  clearInterval(statusTimer);
  statusTimer = setInterval(pollStatus, 700);
}

async function pollStatus() {
  try {
    const state = await (await fetch('/api/state')).json();
    const task = state.task || {};
    $('live-status').textContent = task.active
      ? `${task.stage_label || task.stage || 'running'} — ${task.message || ''}`
      : (task.message || 'Idle.');
  } catch (err) {
    $('live-status').textContent = 'Lost connection to the simulator.';
  }
}

$('live-viewer-close').addEventListener('click', () => {
  $('live-viewer').hidden = true;
  clearInterval(statusTimer);
});

// --- expert seed gallery ---
async function loadGallery() {
  const grid = $('seed-grid');
  let entries;
  try {
    entries = await (await fetch('/media/expert/manifest.json')).json();
  } catch (err) {
    grid.innerHTML = '<p class="loading-note">Could not load the expert gallery manifest.</p>';
    return;
  }
  grid.innerHTML = '';
  for (const entry of entries) {
    const card = document.createElement('div');
    card.className = 'seed-card';
    card.innerHTML = `
      <video src="/media/expert/${entry.video}" controls preload="metadata" muted></video>
      <div class="seed-body">
        <h3>Seed ${entry.seed}</h3>
        <p class="changes">${entry.changes || ''}</p>
        <button data-seed="${entry.seed}">Try this seed live</button>
      </div>`;
    card.querySelector('button').addEventListener('click', async (event) => {
      const seed = Number(event.target.dataset.seed);
      event.target.disabled = true;
      try {
        await api('/api/reset', { scenario: 'dinner', seed, dinner_preset: 'task' });
        await api('/api/task', { action: 'start', kind: 'set_table' });
        showLiveViewer(`Live — seed ${seed}`);
      } catch (err) {
        alert('Could not start the live run: ' + err.message);
      } finally {
        event.target.disabled = false;
      }
    });
    grid.appendChild(card);
  }
}
loadGallery();

// --- generalization: random seed the gallery never showed ---
$('randomize-btn').addEventListener('click', async () => {
  const seed = Math.floor(Math.random() * 2_000_000_000) + 1;
  $('randomize-btn').disabled = true;
  try {
    await api('/api/reset', { scenario: 'dinner', seed, dinner_preset: 'task' });
    $('generalize-seed').textContent = `Seed ${seed} (not in the gallery above)`;
    $('generalize-seed').dataset.seed = seed;
    $('arrange-btn').disabled = false;
    showLiveViewer(`Live — random seed ${seed}`);
  } catch (err) {
    alert('Could not randomize the table: ' + err.message);
  } finally {
    $('randomize-btn').disabled = false;
  }
});

$('arrange-btn').addEventListener('click', async () => {
  $('arrange-btn').disabled = true;
  try {
    await api('/api/task', { action: 'start', kind: 'set_table' });
    showLiveViewer(`Live — seed ${$('generalize-seed').dataset.seed}`);
  } catch (err) {
    alert('Could not start the arrangement task: ' + err.message);
  }
});

// --- live VLA ---
$('vla-run').addEventListener('click', async () => {
  const text = $('vla-instruction').value.trim();
  if (!text) return;
  $('vla-run').disabled = true;
  $('vla-status').textContent = 'Loading the policy and running inference — the first call can take a while on CPU...';
  try {
    await api('/api/command', { text, mode: 'learned_vla' });
    showLiveViewer('Live SmolVLA');
  } catch (err) {
    $('vla-status').textContent = 'Could not start: ' + err.message;
  } finally {
    $('vla-run').disabled = false;
  }
});
