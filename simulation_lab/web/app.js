const $ = (id) => document.getElementById(id);
import {initSpeech} from './speech.js';
$('command-mode').addEventListener('change',()=>{
  const mode=$('command-mode').value;
  $('command-feedback').textContent=mode==='programmed'
    ? 'Programmed mode: set the table, place individual items, or pass the bottle to the right arm.'
    : mode==='learned_dinner_wide'
      ? 'Try “place the bottle”, “place the bottle at the right spot”, or “set the table”. Choose Wider region at reset and change the seed to vary the bottle position. Live vision guides its grasp; other objects retain their narrower trained starts.'
    : mode==='learned_dinner_visual'
      ? 'Learned dinner with live mug vision: say “set the table”. Stereo cameras correct late mug placement. Other skills use initial camera observations and motor feedback. Choose the original dinner controller for individual mug commands. Starting regions and destinations remain limited.'
    : mode==='learned_dinner'
      ? 'Learned dinner candidate: say “set the table” or name an item. Camera checks plan preparation steps; each neural skill releases and parks before the next. Relay instructions require the corresponding trained models. Custom destinations are unsupported.'
    : mode==='learned_bottle'
      ? 'Learned upright mode: reset to Task start, then say or type “place the bottle”. Other objects and custom destinations are not supported.'
      : 'Original learned model: use familiar sideways bottle practice, then say or type “place the bottle”. Wider placements can fail.';
});
initSpeech({button:$('voice-command'),feedback:$('command-feedback'),onTranscript:async text=>{
  $('command-text').value=text;
  // Only a final transcript is executed, never a changing partial hypothesis.
  try{const next=await api('/api/command',{text,mode:$('command-mode').value});updateState(next);$('command-feedback').textContent='Speechmatics: '+text+' — '+next.task.message;}
  catch(error){$('command-feedback').textContent='Speechmatics: '+text+' — '+error.message;}
}});
$('command-form').addEventListener('submit',async event=>{
  event.preventDefault();$('send-command').disabled=true;
  try {const next=await api('/api/command',{text:$('command-text').value,mode:$('command-mode').value});updateState(next);$('command-feedback').textContent='Instruction accepted. '+(next.task.message||'');}
  catch(error){$('command-feedback').textContent=error.message;}
  finally{$('send-command').disabled=false;}
});
let state, selectedArm = 'left', connected = false, controlsReady = false;
let jointTimer, noticeTimer, streamRetry, lastJointEdit = 0, goalBusy = false, slotOptionsKey = '';

function notice(message) {
  $('notice').textContent = message;
  $('notice').hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { $('notice').hidden = true; }, 6500);
}

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {cache: 'no-store'} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) {
    const message = Array.isArray(data.detail) ? data.detail[0].msg : data.detail;
    throw new Error(message || 'The request could not be completed.');
  }
  return data;
}

function updateState(next) {
  const previousVersion = state?.scene_version;
  state = next;
  const dinner = state.scenario === 'dinner';
  $('runtime-mode').textContent=state.task.visual_feedback_profile|| (state.controller==='learned_dinner'?'Neural dinner skills · camera-grounded sequence':state.controller==='learned_bottle'?'OpenVINO neural bottle policy · physical success monitor':state.task.active?'Programmed physical skills · simulator positions':'Runs on this PC · Manual control');
  $('learned-dinner-option').disabled=!state.learned_dinner_available;
  $('visual-mug-option').disabled=!state.visual_mug_available;
  $('wide-bottle-option').disabled=!state.wide_bottle_available;
  $('wide-bottle-start-option').disabled=!state.wide_bottle_available;
  $('policy-description').textContent=state.wide_bottle_available
    ? 'Wider bottle vision supports a larger starting region and two destinations. Use “place the bottle”, its “right spot”, or full table setting. The remaining objects still use their trained starting regions; mug placement has live visual correction.'
    : state.visual_mug_available
    ? 'Live mug vision adds stereo correction during mug placement. Other learned skills use initial camera observations and motor feedback. The original dinner suite and programmed controls remain available.'
    : state.learned_dinner_available
    ? 'The dinner candidate combines six neural skills with camera-based preparation checks. Bottle-only models remain available. Programmed skills use simulator positions.'
    : 'The upright model uses overhead bottle localization and motor feedback. The original model uses three initial views and supports familiar sideways practice. Programmed skills use simulator positions.';
  connected = true;
  $('status-dot').className = 'status-dot connected';
  $('connection-text').textContent = 'Connected to this PC';
  $('engine-label').textContent = state.engine;
  $('shadows').checked = state.shadows;
  $('camera-name').textContent = state.camera_label;
  $('camera-frame').alt = `${state.camera_label}: two SO-101 robot arms and ${dinner ? 'a dinner table with dishes, vessels and a cutlery drawer' : `${state.rack_count} test-tube racks`}`;
  $('scene-label').textContent = `Seed ${state.seed} · ${dinner ? `${state.dinner_preset === 'reference' ? 'Target example' : 'Task start'} · ${state.object_count} items` : `${state.tube_count} tubes`}`;
  $('sim-time').textContent = `t = ${state.simulation_time_s.toFixed(2)} s`;
  $('fps').textContent = state.fps.toFixed(1);
  $('item-count-label').textContent = dinner ? 'Tableware items' : 'Racks';
  $('upright-count-label').textContent = dinner ? 'Upright / above table' : 'Tubes upright';
  $('rack-total').textContent = dinner ? state.object_count : state.rack_count;
  $('upright-total').textContent = dinner ? `${state.stable_object_count} / ${state.object_count}` : `${state.upright_count} / ${state.tube_count}`;
  $('contact-total').textContent = state.contacts;
  $('real-time-factor').textContent = `${state.real_time_factor.toFixed(2)}×`;
  $('control-mode').textContent = state.task.active ? 'Autonomous goal' : 'Manual control';
  $('play-pause').disabled = false;
  $('play-pause').textContent = state.running ? 'Pause' : 'Resume';
  $('step').disabled = state.running;
  $('live-pill').textContent = state.running ? (state.task.active ? `GOAL · ${state.task.tube_id || 'PLANNING'}` : state.motion_test ? 'MOTION TEST' : 'LIVE') : 'PAUSED';
  $('live-pill').className = state.running ? 'live-pill' : 'live-pill paused';
  document.querySelectorAll('[data-camera]').forEach(button => {
    const active = button.dataset.camera === state.camera;
    button.classList.toggle('selected', active);
    button.setAttribute('aria-pressed', active);
  });
  if (!controlsReady) {
    makeJointControls(); syncSceneControls(); controlsReady = true;
    if (dinner && state.task.status !== 'idle') $('dinner-goal').value = state.task.kind === 'set_table' ? 'set_table' : state.task.kind === 'drawer_open' ? 'drawer' : state.task.object_id || 'bottle';
    if (state.task.status !== 'idle' && !dinner) {
      $('goal-kind').value = state.task.kind;
      $('goal-arm').value = state.task.arm || 'auto';
      $('goal-tube').value = state.task.tube_id || '';
      $('goal-destination').value = state.task.destination_slot || '';
    }
  }
  else if (previousVersion !== state.scene_version) syncSceneControls();
  if (Date.now() - lastJointEdit > 600) syncJointControls();
  const p = state.arms[selectedArm].tool_position_m;
  $('tool-position').textContent = `X ${p[0].toFixed(3)}  Y ${p[1].toFixed(3)}  Z ${p[2].toFixed(3)}`;
  updateTask();
  syncGoalSlots();
  if (dinner) updateDinner();
}

function updateDinner() {
  $('drawer-state').textContent = `${Math.max(0, state.drawer.open_m * 100).toFixed(1)} / ${(state.drawer.travel_m * 100).toFixed(1)} cm open`;
  for (const item of state.objects) {
    const row = document.querySelector(`[data-object="${item.id}"]`);
    if (!row) continue;
    row.querySelector('.object-position').textContent = `X ${(item.position_m[0]*100).toFixed(1)} · Y ${(item.position_m[1]*100).toFixed(1)} cm`;
    row.querySelector('.object-condition').textContent = !item.above_table ? 'Below table' : item.upright ? 'Upright' : 'Tilted';
  }
}

function sceneFormMode() {
  const dinner = $('scenario').value === 'dinner';
  $('rack-count-wrap').hidden = dinner;
  $('dinner-preset-wrap').hidden = !dinner;
  $('drawer-start-wrap').hidden = !dinner;
  const reference = dinner && $('dinner-preset').value === 'reference';
  $('bottle-start-wrap').hidden = !dinner;
  $('bottle-start').disabled = reference || !dinner;
  $('drawer-start').disabled = reference;
  $('drawer-start').title = reference ? 'The target example keeps the drawer closed to clear the glass setting.' : '';
  if (reference) { $('drawer-start').value = 'closed'; $('bottle-start').value = 'upright'; }
}

function updateTask() {
  const task = state.task, metrics = task.metrics;
  if (state.scenario === 'dinner') {
    const learned=task.kind==='learned_bottle';
    const dinnerLearned=task.policy_mode==='learned_dinner';
    $('active-controller-label').textContent=dinnerLearned?'Learned dinner · '+(task.policy_details?.neural_runtime||'neural model'):learned?'Learned bottle · OpenVINO':'Physical skills · exact simulator state';
    $('active-controller-description').textContent=dinnerLearned?'Camera observations condition each neural skill. Motor feedback regulates movement; physical checks require release and parked arms before chaining.':learned?(task.policy_details?.visual_encoder==='bottle_rgb_geometry'?'Initial overhead bottle localization conditions a learned trajectory. Motor feedback controls progress; an independent physical monitor checks success.':'Three initial camera views condition the original neural trajectory. Motor feedback controls progress; an independent physical monitor checks success.'):'Move the bottle, plate and mug, open the drawer, then place the fork and spoon. Each grasp and placement is checked.';
    $('dinner-status').textContent = task.status.toUpperCase();
    $('dinner-status').className = `goal-status ${task.status}`;
    $('dinner-stage').textContent = task.status === 'succeeded' ? 'Dinner goal complete' : task.stage_label;
    $('dinner-elapsed').textContent = `${task.elapsed_s.toFixed(1)} s`;
    $('dinner-progress').value = task.progress;
    $('dinner-message').textContent = task.message;
    $('start-dinner').disabled = task.active || goalBusy || !connected || state.dinner_preset !== 'task' || ($('dinner-goal').value === 'set_table' && state.drawer.open_m > .008);
    $('cancel-dinner').disabled = !task.active || goalBusy || !connected;
    $('dinner-goal').disabled = task.active || goalBusy;
    const resultsKey = JSON.stringify((task.results || []).map(r => [r.skill,r.status]));
    if ($('dinner-results').dataset.key !== resultsKey) {
      $('dinner-results').dataset.key = resultsKey;
      $('dinner-results').replaceChildren(...(task.results || []).map(r => {
        const li = document.createElement('li');
        const error=r.metrics?.placement_error_mm??r.metrics?.placement_xy_error_mm;
        li.textContent = `${r.skill_label || r.skill}: ${r.status==='succeeded'?'verified':r.status}${error === undefined ? '' : `, ${error.toFixed(1)} mm placement error`}`;
        return li;
      }));
    }
  }
  const labels = {idle: 'READY', running: state.running ? 'RUNNING' : 'PAUSED', succeeded: 'SUCCESS', failed: 'FAILED', cancelled: 'CANCELLED'};
  $('goal-status').textContent = labels[task.status];
  $('load-sideways').disabled = task.active || goalBusy || !connected;
  $('goal-status').className = `goal-status ${task.status}`;
  $('goal-stage').textContent = task.stage_label;
  $('goal-elapsed').textContent = `${task.elapsed_s.toFixed(1)} s`;
  $('goal-progress').value = task.progress;
  $('task-strip').hidden = task.status === 'idle';
  $('task-strip-stage').textContent = `${labels[task.status]} · ${task.tube_id ? `${state.scenario === 'dinner' ? 'Object' : 'Tube'} ${task.tube_id}${task.destination_slot ? ` → ${task.destination_slot}` : ''} · ` : ''}${task.status === 'succeeded' ? 'Goal complete' : task.stage_label}`;
  $('task-strip-metrics').textContent = `${state.scenario === 'dinner' ? 'Peak lift' : 'Lift'} ${((state.scenario === 'dinner' ? metrics.max_lift_cm : metrics.lift_cm) || 0).toFixed(1)} cm · Hold ${(metrics.hold_verified_s || 0).toFixed(1)} s · ${task.elapsed_s.toFixed(1)} s elapsed`;
  $('task-strip-cancel').hidden = !task.active;
  $('task-strip-cancel').disabled = goalBusy || !connected;
  if ($('goal-message').textContent !== task.message) $('goal-message').textContent = task.message;
  $('goal-lift').textContent = metrics.lift_cm === undefined ? '—' : `${metrics.lift_cm.toFixed(1)} cm`;
  $('goal-hold').textContent = metrics.hold_verified_s === undefined ? '—' : `${metrics.hold_verified_s.toFixed(1)} s`;
  $('practice-label').textContent = `${state.transfer_side ? `Transfer practice · ${state.transfer_side} arm` : state.practice ? 'Lift practice · flat-bottom tubes, 22 mm guides' : 'Randomized scene'} · ${state.tube_count} tubes`;
  $('start-goal').textContent = $('goal-kind').value === 'transfer' ? 'Start transfer' : 'Start lift & return';
  $('goal-destination-wrap').hidden = $('goal-kind').value !== 'transfer';
  $('start-goal').disabled = state.scenario === 'dinner' || task.active || goalBusy || !connected || ($('record-demo').checked && state.recording?.busy);
  $('cancel-goal').disabled = !task.active || goalBusy || !connected;
  ['load-practice', 'load-transfer', 'goal-kind', 'goal-destination', 'record-demo', 'goal-arm', 'goal-tube', 'home-arms', 'gentle-test'].forEach(id => { $(id).disabled = task.active || goalBusy; });
  document.querySelectorAll('#joint-controls input').forEach(input => { input.disabled = task.active; });
  const key = `${task.kind}/${task.status}/${task.stage}`;
  if ($('goal-stages').dataset.state !== key) {
    $('goal-stages').dataset.state = key;
    const activeIndex = task.stages.findIndex(stage => stage.id === task.stage);
    $('goal-stages').replaceChildren(...task.stages.map((stage, i) => {
      const li = document.createElement('li'); li.textContent = stage.label;
      if (task.status === 'succeeded' || i < activeIndex) li.className = 'complete';
      if (i === activeIndex && task.status !== 'succeeded') { li.className = 'current'; li.setAttribute('aria-current', 'step'); }
      return li;
    }));
  }
  $('goal-result').textContent = metrics.placement_xy_error_mm === undefined ? (task.tube_id ? `${task.arm} arm · tube ${task.tube_id} · ${task.source_slot} → ${task.destination_slot}.` : '') :
    `${task.source_slot} → ${task.destination_slot}. Peak lift ${metrics.max_lift_cm.toFixed(1)} cm. Placement error ${metrics.placement_xy_error_mm.toFixed(1)} mm; tilt ${metrics.tilt_deg.toFixed(1)}°. Parked arm motion ${metrics.other_arm_max_motion_deg.toFixed(2)}°.`;
  const recording = state.recording || {status:'idle'};
  $('recording-status').textContent = recording.message || 'No demonstration recorded yet.';
  $('recording-counts').textContent = recording.id ? `${recording.observations || 0} observations · ${recording.actions || 0} actions · ${recording.images || 0} images` : '';
  $('recording-link').hidden = !recording.id || recording.busy;
  if (recording.id) $('recording-link').href = `/api/recordings/${encodeURIComponent(recording.id)}/manifest`;
}

function syncGoalSlots() {
  const key = JSON.stringify([state.tubes.map(t => [t.id,t.slot]), state.slots?.map(s => [s.id,s.blocked_by])]);
  if (key === slotOptionsKey) return;
  slotOptionsKey = key;
  const tube = $('goal-tube').value, destination = $('goal-destination').value;
  $('goal-tube').replaceChildren(new Option('Choose automatically', ''), ...state.tubes.map(t => new Option(`${t.id} · ${t.slot ? `in ${t.slot}` : 'outside a slot'}`, t.id)));
  if (state.tubes.some(t => t.id === tube)) $('goal-tube').value = tube;
  $('goal-destination').replaceChildren(new Option('Choose an empty slot', ''), ...(state.slots || []).map(slot => {
    const option = new Option(`${slot.id} · ${slot.blocked_by.length ? `occupied / blocked (${slot.blocked_by.join(', ')})` : 'empty'}`, slot.id);
    return option;
  }));
  if (state.slots?.some(s => s.id === destination)) $('goal-destination').value = destination;
  for (const option of $('rack-select').options) {
    const rack = state.racks.find(r => r.id === option.value);
    if (rack) option.textContent = `Rack ${rack.id} · ${rack.slots.length} tubes`;
  }
  syncRackSummary();
}

function makeJointControls() {
  $('joint-controls').replaceChildren();
  state.arms[selectedArm].joints.forEach((joint, index) => {
    const group = document.createElement('div');
    group.className = 'joint';
    const label = document.createElement('label');
    label.className = 'range-label';
    label.htmlFor = `joint-${index}`;
    const name = document.createElement('span'); name.textContent = joint.label;
    const output = document.createElement('output'); output.id = `joint-value-${index}`;
    label.append(name, output);
    const input = document.createElement('input');
    Object.assign(input, {id: `joint-${index}`, type: 'range', min: joint.min_deg, max: joint.max_deg, step: '0.5', value: joint.target_deg});
    input.addEventListener('input', () => {
      lastJointEdit = Date.now();
      output.textContent = `${Number(input.value).toFixed(1)}°`;
      clearTimeout(jointTimer);
      jointTimer = setTimeout(sendJoints, 90);
    });
    const limits = document.createElement('div'); limits.className = 'joint-limits';
    limits.innerHTML = `<span>${Math.round(joint.min_deg)}°</span><span>${Math.round(joint.max_deg)}°</span>`;
    group.append(label, input, limits);
    $('joint-controls').append(group);
  });
  syncJointControls();
}

function syncJointControls() {
  state.arms[selectedArm].joints.forEach((joint, i) => {
    $(`joint-${i}`).value = joint.target_deg;
    $(`joint-value-${i}`).textContent = `${joint.target_deg.toFixed(1)}°`;
  });
}

async function sendJoints() {
  const targets = Array.from({length: 6}, (_, i) => Number($(`joint-${i}`).value));
  await control({arm: selectedArm, targets_deg: targets});
}

async function control(payload) {
  try { updateState(await api('/api/control', payload)); }
  catch (error) { notice(error.message); }
}

function syncSceneControls() {
  const dinner = state.scenario === 'dinner';
  $('scenario').value = state.scenario || 'chemistry';
  $('dinner-panel').hidden = !dinner;
  $('chemistry-goals').hidden = dinner;
  $('rack-tab').hidden = dinner;
  if (dinner) {
    $('dinner-preset').value = state.dinner_preset;
    $('drawer-start').value = state.drawer_open ? 'open' : 'closed';
    $('bottle-start').value=state.bottle_start||'upright';
    document.querySelector('[data-panel="arms"]').click();
    $('object-inventory').replaceChildren(...state.objects.map(item => {
      const row = document.createElement('div'); row.className = 'object-row'; row.dataset.object = item.id;
      const label = document.createElement('strong'); label.textContent = item.label;
      const swatch = document.createElement('i'); swatch.className = 'swatch'; swatch.style.background = item.color;
      label.prepend(swatch);
      const position = document.createElement('span'); position.className = 'object-position';
      const condition = document.createElement('small'); condition.className = 'object-condition';
      row.append(label, position, condition); return row;
    }));
  }
  sceneFormMode();
  $('viewer-note').textContent = dinner ? 'Physical tableware and a passive drawer. The target example is a reset preset, not an autonomous result.' : 'A rigid-body learning scene. Tubes have gravity and collisions; the colored contents are visual markers.';
  $('seed').value = state.seed;
  if (!dinner) $('rack-count').value = state.rack_count;
  slotOptionsKey = ''; syncGoalSlots();
  const selected = $('rack-select').value;
  $('rack-select').replaceChildren(...state.racks.map(rack => {
    const option = document.createElement('option');
    option.value = rack.id; option.textContent = `Rack ${rack.id} · ${rack.slots.length} tubes`;
    return option;
  }));
  if (state.racks.some(rack => rack.id === selected)) $('rack-select').value = selected;
  syncRackEditor();
  syncRackSummary();
}

function syncRackSummary() {
  $('rack-summary').replaceChildren(...state.racks.map(rack => {
    const item = document.createElement('div'); item.className = 'rack-item';
    item.innerHTML = `<span class="rack-name"><i class="swatch" style="background:${rack.color}"></i>Rack ${rack.id}</span><small>${rack.slots.length} tubes · ${Math.round(rack.yaw_deg)}°</small>`;
    return item;
  }));
}

function syncRackEditor() {
  const rack = state.racks.find(r => r.id === $('rack-select').value) || state.racks[0];
  if (!rack) return;
  $('rack-x').value = rack.x * 100;
  $('rack-y').value = rack.y * 100;
  $('rack-yaw').value = rack.yaw_deg;
  updateRackOutputs();
}

function updateRackOutputs() {
  $('rack-x-value').textContent = `${Number($('rack-x').value).toFixed(1)} cm`;
  $('rack-y-value').textContent = `${Number($('rack-y').value).toFixed(1)} cm`;
  $('rack-yaw-value').textContent = `${Math.round(Number($('rack-yaw').value))}°`;
}

async function resetScene(seed, rackCount) {
  clearTimeout(jointTimer);
  try {
    $('loading').hidden = false;
    updateState(await api('/api/reset', {seed, rack_count: rackCount, scenario: $('scenario').value,
      dinner_preset: $('dinner-preset').value, drawer_open: $('drawer-start').value === 'open',
      bottle_start:$('scenario').value==='dinner'&&$('dinner-preset').value==='task'?$('bottle-start').value:'upright'}));
    syncSceneControls();
  } catch (error) { notice(error.message); }
  finally { $('loading').hidden = true; }
}

async function sendGoal(action) {
  clearTimeout(jointTimer);
  goalBusy = true; updateTask();
  try {
    let payload = {action};
    let endpoint='/api/task';
    if (action === 'start' && state.scenario === 'dinner') {
      const selected = $('dinner-goal').value;
      if($('command-mode').value!=='programmed') {
        endpoint='/api/command';
        payload={mode:$('command-mode').value,text:selected==='set_table'?'set the table':selected==='drawer'?'open the drawer':`place the ${selected}`};
      } else {
        payload = {action, kind: selected === 'set_table' ? 'set_table' : selected === 'drawer' ? 'drawer_open' : 'dinner_place',
          object_id: ['set_table','drawer'].includes(selected) ? null : selected, arm:'auto', record:false};
      }
    } else if (action === 'start') {
      const kind = $('goal-kind').value;
      payload = {action, kind, arm: $('goal-arm').value, tube_id: $('goal-tube').value || null,
        destination_slot: kind === 'transfer' ? $('goal-destination').value || null : null, record: $('record-demo').checked};
    }
    updateState(await api(endpoint, payload));
    if (action === 'start' && window.matchMedia('(max-width:790px)').matches) $('camera-frame').scrollIntoView({behavior:'smooth', block:'start'});
  } catch (error) { notice(error.message); }
  finally { goalBusy = false; updateTask(); }
}
$('dinner-goal').addEventListener('change', () => state && updateTask());
$('start-dinner').addEventListener('click', () => sendGoal('start'));
$('cancel-dinner').addEventListener('click', () => sendGoal('cancel'));
$('start-goal').addEventListener('click', () => sendGoal('start'));
$('cancel-goal').addEventListener('click', () => sendGoal('cancel'));
$('task-strip-cancel').addEventListener('click', () => sendGoal('cancel'));
$('goal-kind').addEventListener('change', () => state && updateTask());
$('record-demo').addEventListener('change', () => state && updateTask());
$('load-sideways').addEventListener('click', async () => {
  if (state?.task?.active) return;
  try {
    updateState(await api('/api/reset', {scenario:'dinner', seed:42, dinner_preset:'task', bottle_start:'sideways'}));
    $('dinner-goal').value = 'bottle';
  } catch (error) { notice(error.message); }
});
$('load-practice').addEventListener('click', async () => {
  clearTimeout(jointTimer); goalBusy = true;
  if (state) updateTask();
  try {
    updateState(await api('/api/reset', {seed: Number($('seed').value), rack_count: 2, practice: true}));
    $('goal-kind').value = 'lift_return'; $('goal-destination').value = '';
    $('goal-tube').value = ''; syncSceneControls();
  } catch (error) { notice(error.message); }
  finally { goalBusy = false; if (state) updateTask(); }
});
$('load-transfer').addEventListener('click', async () => {
  clearTimeout(jointTimer); goalBusy = true;
  if (state) updateTask();
  try {
    const side = $('goal-arm').value === 'right' ? 'right' : 'left';
    updateState(await api('/api/reset', {seed: Number($('seed').value), rack_count: 2, transfer_side: side}));
    $('goal-kind').value = 'transfer'; $('goal-arm').value = side; $('goal-tube').value = 'A2'; $('goal-destination').value = 'B2';
    syncSceneControls();
  } catch (error) { notice(error.message); }
  finally { goalBusy = false; if (state) updateTask(); }
});

$('play-pause').addEventListener('click', () => state && control({running: !state.running}));
$('step').addEventListener('click', () => control({step: true}));
$('shadows').addEventListener('change', () => control({shadows: $('shadows').checked}));
$('home-arms').addEventListener('click', () => { lastJointEdit = 0; control({preset: 'home'}); });
$('gentle-test').addEventListener('click', () => { lastJointEdit = 0; control({preset: 'gentle'}); });
$('reset-scene').addEventListener('click', async () => {
  if (!state) return;
  clearTimeout(jointTimer);
  if (state.scenario === 'dinner') {
    try { updateState(await api('/api/reset', {scenario: 'dinner', seed: state.seed, dinner_preset: state.dinner_preset, drawer_open: state.drawer_open,bottle_start:state.bottle_start||'upright'})); }
    catch (error) { notice(error.message); }
    return;
  }
  const racks = state.racks.map(({x, y, yaw_deg}) => ({x, y, yaw_deg}));
  try { updateState(await api('/api/racks', {racks})); syncSceneControls(); }
  catch (error) { notice(error.message); }
});
$('seed-form').addEventListener('submit', event => { event.preventDefault(); resetScene(Number($('seed').value), Number($('rack-count').value)); });
$('scenario').addEventListener('change', sceneFormMode);
$('dinner-preset').addEventListener('change', sceneFormMode);
$('shuffle').addEventListener('click', () => { const n = new Uint32Array(1); crypto.getRandomValues(n); resetScene(n[0] % 2147483647, Number($('rack-count').value)); });
document.querySelectorAll('[data-camera]').forEach(button => button.addEventListener('click', () => control({camera: button.dataset.camera})));
document.querySelectorAll('[data-arm]').forEach(button => button.addEventListener('click', () => {
  if (!state) return;
  clearTimeout(jointTimer);
  selectedArm = button.dataset.arm; lastJointEdit = 0;
  document.querySelectorAll('[data-arm]').forEach(b => { b.classList.toggle('selected', b === button); b.setAttribute('aria-pressed', b === button); });
  makeJointControls();
}));
document.querySelectorAll('[data-panel]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('[data-panel]').forEach(b => { b.classList.toggle('selected', b === button); b.setAttribute('aria-pressed', b === button); });
  $('arms-panel').hidden = button.dataset.panel !== 'arms';
  $('racks-panel').hidden = button.dataset.panel !== 'racks';
}));
$('rack-select').addEventListener('change', () => state && syncRackEditor());
['rack-x', 'rack-y', 'rack-yaw'].forEach(id => $(id).addEventListener('input', updateRackOutputs));
$('apply-rack').addEventListener('click', async () => {
  if (!state) return;
  const racks = state.racks.map(rack => rack.id === $('rack-select').value ? {x: Number($('rack-x').value) / 100, y: Number($('rack-y').value) / 100, yaw_deg: Number($('rack-yaw').value)} : {x: rack.x, y: rack.y, yaw_deg: rack.yaw_deg});
  try { updateState(await api('/api/racks', {racks})); syncSceneControls(); }
  catch (error) { notice(error.message); }
});

function connectStream() {
  clearTimeout(streamRetry);
  $('camera-frame').src = `/stream?connection=${Date.now()}`;
}
$('camera-frame').addEventListener('load', () => { $('loading').hidden = true; });
$('camera-frame').addEventListener('error', () => { $('loading').hidden = false; streamRetry = setTimeout(connectStream, 3000); });

async function poll() {
  try {
    updateState(await api('/api/state'));
    if (!$('camera-frame').getAttribute('src')) connectStream();
    $('loading').hidden = state.ready === true;
  } catch (error) {
    if (connected) notice(`Simulator disconnected: ${error.message}`);
    connected = false;
    $('status-dot').className = 'status-dot offline';
    $('connection-text').textContent = 'Simulator unavailable';
    $('play-pause').disabled = true; $('step').disabled = true;
    $('start-goal').disabled = true; $('cancel-goal').disabled = true;
  }
  setTimeout(poll, document.hidden ? 2000 : 350);
}
poll();
