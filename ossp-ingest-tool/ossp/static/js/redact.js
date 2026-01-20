(() => {
  document.addEventListener('DOMContentLoaded', () => {
    const form        = document.getElementById('redaction-form');
    const tableBody   = document.getElementById('rules-table-body');
    const rulesInput  = document.getElementById('rules_json');

    const runBtn      = document.getElementById('run-redaction-btn');
    const spinnerSpan = document.getElementById('redact-spinner');
    const downloadBtn = document.getElementById('download-redaction-btn');
    const resetBtn    = document.getElementById('reset-values-btn');
    const presetSelect   = document.getElementById('presetSelect');
    const applyPresetBtn = document.getElementById('applyPresetBtn');
    const presetNameInput      = document.getElementById('presetNameInput');
    const savePresetBtn        = document.getElementById('savePresetBtn');
    const selectedPresetInput  = document.getElementById('selectedPresetInput');

    // In-memory rules list
    let rules = [];


    // Input sanitization - turn input into strings
    const deepEqual = (a, b) => JSON.stringify(a) === JSON.stringify(b);

    function syncHiddenInput() {
      rulesInput.value = JSON.stringify(rules);
    }

    function modeLabel(type) {
      switch (type) {
        case 'byKeyName':        return 'Redact by key/field name';
        case 'byFieldValue':     return 'Redact by field value';
        case 'rowByFieldValue':  return 'Redact entire row by field value';
        case 'multiKeyValue':    return 'Redact when KEY = VALUE';
        default:                 return type;
      }
    }

    function valuesToText(rule) {
      if (rule.type === 'multiKeyValue') {
        // r.values = [ [ Key, value, "f1,f2" ], ... ]
        return rule.values.map(v => `[${v[0]} = ${v[1]} -> ${v[2]}]`).join('; ');
      }
      return rule.values.join(', ');
    }

    function renderTable() {
      tableBody.innerHTML = '';
      rules.forEach((r, i) => {
        const tr = document.createElement('tr');
        tr.dataset.index = i;
        tr.innerHTML = `
          <td>${i + 1}</td>
          <td>${modeLabel(r.type)}</td>
          <td>${r.key || '-'}</td>
          <td>${valuesToText(r)}</td>
          <td>
            <button type="button" class="btn btn-sm btn-danger remove-rule-btn" data-index="${i}">Remove</button>
          </td>`;
        tableBody.appendChild(tr);
      });
    }


    function showSpinner() {
      runBtn.disabled = true;
      spinnerSpan.innerHTML =
        '<i class="bi bi-arrow-repeat me-2 spinner-border spinner-border-sm"></i>Running…';
    }

    function hideSpinner() {
      runBtn.disabled = false;
      spinnerSpan.innerHTML = '<i class="bi bi-play-circle me-2"></i>Run Redaction';
    }


    async function loadPresets() {
      if (!presetSelect) return;
      try {
        const rsp = await fetch('/redact/presets');
        const payload = await rsp.json();
        const presets = payload.presets || [];
        // repopulate dropdown
        const existing = new Set(Array.from(presetSelect.options).map(o => o.value));
        presets.forEach(name => {
          if (!existing.has(name)) {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            presetSelect.appendChild(opt);
          }
        });
      } catch (e) {
        console.error('Failed to load presets', e);
      }
    }

    async function applyPreset(name, opts = {}) {
      // If "None - start blank" is selected (empty value), clear rules instead of fetching
      if (!name) {
        if (rules.length && !opts.silentConfirm) {
          const ok = confirm(`Replace existing rules with policy "${name}"? This will discard the current rules.`);
          if (!ok) return;
        }
        // Clear rules and UI
        rules = [];
        renderTable();
        syncHiddenInput();

        // Clear the hidden preset input so the server sees no policy selected
        if (selectedPresetInput) selectedPresetInput.value = '';

        // Disable download until redaction runs again
        downloadBtn.disabled = true;
        downloadBtn.classList.remove('btn-success');
        downloadBtn.classList.add('btn-secondary');
        return; // Done
      }

      try {
        const rsp = await fetch(`/redact/rules?preset=${encodeURIComponent(name)}`);
        const data = await rsp.json();
        if (!rsp.ok) throw new Error(data.error || 'Failed to load preset rules');

        // Replace in-memory rules and refresh UI
        rules = data.rules || [];
        renderTable();
        syncHiddenInput();

        // Invalidate any previous download until user runs redaction again
        downloadBtn.disabled = true;
        downloadBtn.classList.remove('btn-success');
        downloadBtn.classList.add('btn-secondary');
      } catch (e) {
        console.error('Apply preset failed:', e);
        alert(`Apply preset failed:\n${e.message}`);
      }
    }

    function syncPresetHidden() {
      if (selectedPresetInput && presetSelect) {
        selectedPresetInput.value = presetSelect.value || '';
      }
    }


    if (applyPresetBtn && presetSelect) {
      applyPresetBtn.addEventListener('click', () => applyPreset(presetSelect.value));
    }

    if (savePresetBtn) {
      savePresetBtn.addEventListener('click', async () => {
        const name = (presetNameInput.value || '').trim();
        if (!name) return alert('Please enter a policy name');
        if (name.toLowerCase() === 'default') {
          return alert('Saving to "default" is not allowed. Pick a different name.');
        }

        // Build payload
        syncHiddenInput();
        const fd = new FormData();
        fd.append('name', name);
        fd.append('rules_json', rulesInput.value);

        try {
          const rsp = await fetch('/redact/save', { method: 'POST', body: fd });
          const payload = await rsp.json().catch(() => ({}));
          if (!rsp.ok) throw new Error(payload.error || payload.message || 'Unknown error');

          // Add to dropdown and select it
          if (presetSelect && !Array.from(presetSelect.options).some(o => o.value === name)) {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            presetSelect.appendChild(opt);
          }
          presetSelect.value = name;
          if (selectedPresetInput) selectedPresetInput.value = name;

          alert('Policy saved');
          // Clear the input box after a successful save
          presetNameInput.value = '';
          // optional: remove focus so the placeholder shows again
          presetNameInput.blur();
        } catch (e) {
          console.error('Save policy failed:', e);
          alert('Save policy failed:\n' + e.message);
        }
      });
    }


    // Add Rule
    document.querySelectorAll('.add-rule-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const type = btn.dataset.type;
        const keySelect = document.querySelector(btn.dataset.keySelect);
        const valueInput = btn.dataset.valueInput ? document.querySelector(btn.dataset.valueInput) : null;
        const fieldsSelect = btn.dataset.fieldsSelect ? document.querySelector(btn.dataset.fieldsSelect) : null;

        let key = null;
        let values = [];

        if (type === 'byKeyName') {
          const selected = Array.from(keySelect.selectedOptions).map(o => o.value);
          if (!selected.length) {
            return alert('Please select at least one field name to redact');
          }
          values = selected;
        } else if (type === 'byFieldValue') {
          const fieldKey = keySelect.value;
          const fieldVal = (valueInput?.value || '').trim();
          if (!fieldKey) return alert('Please select a field name');
          if (!fieldVal) return alert('Please enter a value or regex');
          key = fieldKey;
          values = [fieldVal];
        } else if (type === 'rowByFieldValue') {
          const rowKey = keySelect.value;
          const rowVal = (valueInput?.value || '').trim();
          if (!rowKey) return alert('Please select a field name');
          if (!rowVal) return alert('Please enter a value or regex');
          key = rowKey;
          values = [rowVal];
        } else if (type === 'multiKeyValue') {
          const triggerKey = keySelect.value;
          const triggerVal = (valueInput?.value || '').trim();
          const redactFields = Array.from(fieldsSelect.selectedOptions).map(o => o.value);
          if (!triggerKey) return alert('Please pick a key');
          if (!triggerVal) return alert('Please enter a value or regex');
          if (!redactFields.length) return alert('Please select at least one field to redact');
          key = triggerKey;
          values = [[triggerKey, triggerVal, redactFields.join(',')]];
        } else {
          return console.error('Unknown rule type', type);
        }

        // Duplicate detection
        const isDuplicate = rules.some(r =>
          r.type === type &&
          r.key === key &&
          deepEqual(r.values, values)
        );
        if (isDuplicate) {
          return alert('This rule already exists');
        }

        rules.push({ type, key, values });

        // Clear inputs for convenience
        if (keySelect?.tagName?.toLowerCase() === 'select') {
          if (keySelect.multiple) {
            Array.from(keySelect.options).forEach(o => (o.selected = false));
          } else {
            keySelect.selectedIndex = 0; // back to "Select Key"
          }
        }
        if (valueInput) valueInput.value = '';
        if (fieldsSelect && fieldsSelect.multiple) {
          Array.from(fieldsSelect.options).forEach(o => (o.selected = false));
        }

        renderTable();
        syncHiddenInput();
      });
    });

    // Remove Rule
    tableBody.addEventListener('click', evt => {
      const btn = evt.target.closest('.remove-rule-btn');
      if (!btn) return;
      const idx = parseInt(btn.dataset.index, 10);
      rules.splice(idx, 1);
      renderTable();
      syncHiddenInput();
    });

    // Run Redaction via fetch with spinner and stay on page
    runBtn.addEventListener('click', async evt => {
      evt.preventDefault();
      syncHiddenInput();
      syncPresetHidden();

      showSpinner();
      try {
        const fd = new FormData(form);
        const rsp = await fetch(form.action, { method: form.method, body: fd });
        const payload = await rsp.json().catch(() => ({}));

        if (!rsp.ok) {
          const msg = payload.error || payload.message || 'Unknown error';
          throw new Error(msg);
        }

        // Success, keep page, enable download
        downloadBtn.disabled = false;
        downloadBtn.classList.remove('btn-secondary');
        downloadBtn.classList.add('btn-success');
        // Optional toast or alert
        // alert(payload.message || 'Redaction successful');
      } catch (err) {
        console.error('Redaction failed:', err);
        alert('Redaction failed:\n' + err.message);
      } finally {
        hideSpinner();
      }
    });

    // Download redacted zip, stay on page
    downloadBtn.addEventListener('click', evt => {
      evt.preventDefault();
      window.location.href = '/download/redacted';
    });

    // Reset
    resetBtn.addEventListener('click', (evt) => {
    evt.preventDefault();

    // Clear rules and UI
    rules.length = 0;          // clear in place
    renderTable();
    syncHiddenInput();

    // Uncheck all checkboxes and disable mapped inputs
    document.querySelectorAll('.form-check-input').forEach(cb => {
      cb.checked = false;

      const targets = toggleMap[cb.id] || (cb.dataset.target ? [cb.dataset.target] : []);
      setEnabled(targets, false);
    });

    // Close all accordion panels
    document.querySelectorAll('.accordion-collapse').forEach(el => {
      const bsColl = (typeof bootstrap !== 'undefined')
        ? bootstrap.Collapse.getOrCreateInstance(el, { toggle: false })
        : null;

      if (bsColl) bsColl.hide();
      else el.classList.remove('show');
    });

    // Reset input contents inside the form for convenience
    const formEl = document.getElementById('redaction-form');
    if (formEl) {
      formEl.querySelectorAll('input[type="text"]').forEach(inp => { inp.value = ''; });
      formEl.querySelectorAll('select').forEach(sel => {
        if (sel.multiple) Array.from(sel.options).forEach(o => o.selected = false);
        else sel.selectedIndex = 0;
      });
    }

    // Disable download button
    downloadBtn.disabled = true;
    downloadBtn.classList.remove('btn-success');
    downloadBtn.classList.add('btn-secondary');
  });
  renderTable();
  syncHiddenInput();
  loadPresets();
  });
})();