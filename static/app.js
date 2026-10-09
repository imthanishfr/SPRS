/*
  SPRS Authentic Industrial Frontend Controller & Fixed Three.js Raycasting Inspector (static/app.js)
*/

let currentStep = 1;
let currentSource = "COM4";

// Three.js Scene Variables
let scene, camera, renderer, controls;
let boxMesh, productMesh, paddingMesh;
let raycaster, mouse;
let currentPackingData = null;


document.addEventListener("DOMContentLoaded", () => {
  initLiveWaveWallpaper();
  init3DVisualizer();
  pollState();
  setInterval(pollState, 800);

  // Global Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      const modal = document.getElementById("calibration-modal");
      if (modal && modal.style.display !== "none") return;
      e.preventDefault();
      handleMainCapture();
    } else if (e.key === "r" || e.key === "R") {
      resetScan();
    }
  });
});

/* ----------------------------------------------------
   0. CAMERA CALIBRATION SETUP MODAL CONTROLLER
---------------------------------------------------- */
function closeCalibrationModal() {
  const modal = document.getElementById("calibration-modal");
  if (modal) modal.style.display = "none";
}

async function submitCalibration() {
  const h = parseFloat(document.getElementById("cal-height-input")?.value || 50.0);
  const refL = parseFloat(document.getElementById("cal-ref-l")?.value || 10.0);

  try {
    const res = await fetch("/api/calibrate_custom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_height_cm: h,
        ref_length_cm: refL
      })
    });
    const data = await res.json();
    if (data.status === "success") {
      console.log("[SPRS Calibration] Calibrated scale:", data.pixels_per_cm, "px/cm");
    }
  } catch (err) {
    console.warn("[SPRS Calibration] Calibration request warning:", err);
  }
  closeCalibrationModal();
}

/* ----------------------------------------------------
   1. LIVE ANIMATED ABSTRACT WAVE WALLPAPER (IMAGE 2)
---------------------------------------------------- */
function initLiveWaveWallpaper() {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let w = (canvas.width = window.innerWidth);
  let h = (canvas.height = window.innerHeight);

  window.addEventListener("resize", () => {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  });

  let mouseX = w / 2;
  let mouseY = h / 2;
  let targetMouseX = w / 2;
  let targetMouseY = h / 2;

  window.addEventListener("mousemove", (e) => {
    targetMouseX = e.clientX;
    targetMouseY = e.clientY;
  });

  let time = 0;

  function render() {
    time += 0.005;

    mouseX += (targetMouseX - mouseX) * 0.04;
    mouseY += (targetMouseY - mouseY) * 0.04;

    ctx.clearRect(0, 0, w, h);

    // 1. Deep Pitch Black Background Base (#030008)
    ctx.fillStyle = "#030008";
    ctx.fillRect(0, 0, w, h);

    // 2. Large Bottom-Curved Radial Glow (Subtle Dark Violet Aura)
    const mOffsetX = (mouseX - w / 2) * 0.12;
    const mOffsetY = (mouseY - h / 2) * 0.08;

    const radGrad = ctx.createRadialGradient(
      w * 0.5 + mOffsetX,
      h * 1.05 + mOffsetY,
      20,
      w * 0.5,
      h * 0.65,
      w * 0.75
    );
    radGrad.addColorStop(0, "rgba(88, 28, 135, 0.22)");   // Soft Violet Glow
    radGrad.addColorStop(0.35, "rgba(46, 16, 101, 0.15)");  // Deep Subtle Purple
    radGrad.addColorStop(0.7, "rgba(23, 7, 46, 0.08)");    // Very Low opacity aura
    radGrad.addColorStop(1, "rgba(3, 0, 8, 0)");

    ctx.fillStyle = radGrad;
    ctx.fillRect(0, 0, w, h);

    // 3. Multi-Layer Animated Sinusoidal Abstract Waves (Subtle Background Effect)
    const waves = [
      {
        gradient: ["rgba(67, 26, 120, 0.15)", "rgba(23, 7, 46, 0.5)"],
        speed: 0.004,
        freq: 0.0014,
        amp: 95,
        baseH: 0.68,
        phase: 0
      },
      {
        gradient: ["rgba(91, 33, 182, 0.12)", "rgba(46, 16, 101, 0.55)"],
        speed: 0.006,
        freq: 0.0022,
        amp: 115,
        baseH: 0.73,
        phase: 2.1
      },
      {
        gradient: ["rgba(124, 58, 237, 0.08)", "rgba(30, 10, 60, 0.6)"],
        speed: 0.003,
        freq: 0.0011,
        amp: 75,
        baseH: 0.79,
        phase: 4.2
      }
    ];

    waves.forEach((wave) => {
      ctx.beginPath();
      ctx.moveTo(0, h);

      const step = Math.max(10, Math.floor(w / 120));
      for (let x = 0; x <= w + step; x += step) {
        const sine1 = Math.sin(x * wave.freq + time * wave.speed * 100 + wave.phase);
        const sine2 = Math.cos(x * 0.0008 + time * 0.4) * (wave.amp * 0.35);
        const mouseMod = Math.sin((x / w) * Math.PI) * ((mouseY - h / 2) * 0.06);

        const y = h * wave.baseH + sine1 * wave.amp + sine2 + mouseMod;
        ctx.lineTo(x, y);
      }

      ctx.lineTo(w, h);
      ctx.closePath();

      const waveGrad = ctx.createLinearGradient(0, h * 0.5, 0, h);
      waveGrad.addColorStop(0, wave.gradient[0]);
      waveGrad.addColorStop(1, wave.gradient[1]);

      ctx.fillStyle = waveGrad;
      ctx.fill();
    });

    requestAnimationFrame(render);
  }

  render();
}

/* ----------------------------------------------------
   2. FIXED THREE.JS 3D VISUALIZER ENGINE (520px PANEL)
---------------------------------------------------- */
function init3DVisualizer() {
  const container = document.getElementById("canvas-3d");
  if (!container) return;

  const w = container.clientWidth || 800;
  const h = container.clientHeight || 520;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);

  camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 1000);
  camera.position.set(38, 28, 48);

  renderer = new THREE.WebGLRenderer({ canvas: container, antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);

  if (typeof THREE.OrbitControls !== "undefined") {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
  }

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
  scene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
  dirLight1.position.set(30, 50, 30);
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.5);
  dirLight2.position.set(-30, 20, -30);
  scene.add(dirLight2);

  // Floor Grid
  const grid = new THREE.GridHelper(80, 24, 0x333333, 0x171717);
  grid.position.y = -0.1;
  scene.add(grid);

  // Raycaster for Click Inspector
  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  container.addEventListener("click", onCanvasClick);

  // Default 3D Box & Item
  update3DScene(25, 20, 15, 17.6, 13.7, 6.2, "BOX-S2", "Electronics", "Anti-Static Bubble Wrap");

  function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener("resize", () => {
    const nw = container.clientWidth;
    const nh = container.clientHeight;
    camera.aspect = nw / nh;
    camera.updateProjectionMatrix();
    renderer.setSize(nw, nh);
  });
}

function update3DScene(boxL, boxW, boxH, objL, objW, objH, boxName = "BOX-S2", catName = "Electronics", fillName = "Bubble Wrap") {
  if (!scene) return;

  currentPackingData = { boxL, boxW, boxH, objL, objW, objH, boxName, catName, fillName };

  if (boxMesh) scene.remove(boxMesh);
  if (productMesh) scene.remove(productMesh);
  if (paddingMesh) scene.remove(paddingMesh);

  // Normalization scale factor
  const maxDim = Math.max(boxL, boxW, boxH, 1.0);
  const scale = 24.0 / maxDim;

  const bL = boxL * scale;
  const bW = boxW * scale;
  const bH = boxH * scale;

  const oL = Math.min(bL * 0.92, objL * scale);
  const oW = Math.min(bW * 0.92, objW * scale);
  const oH = Math.min(bH * 0.92, objH * scale);

  const packGroup = new THREE.Group();
  const centerY = bH / 2.0;

  // 1. CARDBOARD BOX (Outer Translucent Amber/Cardboard Brown + Gold Wireframe)
  const boxGeo = new THREE.BoxGeometry(bL, bH, bW);
  const boxMat = new THREE.MeshPhongMaterial({
    color: 0xc2410c, // Cardboard Amber
    transparent: true,
    opacity: 0.18,
    depthWrite: false,
    side: THREE.DoubleSide
  });
  boxMesh = new THREE.Mesh(boxGeo, boxMat);
  boxMesh.position.set(0, centerY, 0);
  boxMesh.userData = { 
    type: "box", 
    name: `Cardboard Box (${boxName})`, 
    details: `Dimensions: ${boxL} × ${boxW} × ${boxH} cm | Cardboard Grade: Heavy Duty Corrugated (32 ECT)` 
  };

  const edgesGeo = new THREE.EdgesGeometry(boxGeo);
  const edgesMat = new THREE.LineBasicMaterial({ color: 0xf97316, linewidth: 2 });
  const wireframe = new THREE.LineSegments(edgesGeo, edgesMat);
  boxMesh.add(wireframe);

  // 2. VOID FILL PADDING LAYER (Fills ENTIRE Box Volume around suspended product)
  const padL = bL * 0.985;
  const padW = bW * 0.985;
  const padH = bH * 0.985;

  const padGeo = new THREE.BoxGeometry(padL, padH, padW);
  const padMat = new THREE.MeshPhongMaterial({
    color: 0x0284c7, // Sky Blue Translucent Cushion
    transparent: true,
    opacity: 0.32,
    depthWrite: false,
    side: THREE.DoubleSide
  });
  paddingMesh = new THREE.Mesh(padGeo, padMat);
  paddingMesh.position.set(0, centerY, 0);
  
  const voidVol = Math.max(0, Math.round((boxL * boxW * boxH) - (objL * objW * objH)));
  paddingMesh.userData = { 
    type: "infill", 
    name: `Protective Infill Cushion (${fillName})`, 
    details: `Required Void Fill Volume: ${voidVol.toLocaleString()} cm³ | Fills 100% of internal box clearance surrounding item` 
  };

  const padEdgesGeo = new THREE.EdgesGeometry(padGeo);
  const padEdgesMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.6 });
  const padWire = new THREE.LineSegments(padEdgesGeo, padEdgesMat);
  paddingMesh.add(padWire);

  // 3. PRODUCT ITEM (Solid Emerald Green Box centered in ALL AXES: X, Y, Z)
  const prodGeo = new THREE.BoxGeometry(oL, oH, oW);
  const prodMat = new THREE.MeshStandardMaterial({
    color: 0x16a34a, // Emerald Green
    roughness: 0.25,
    metalness: 0.2
  });
  productMesh = new THREE.Mesh(prodGeo, prodMat);
  // Center in ALL AXES inside box
  productMesh.position.set(0, centerY, 0);
  productMesh.userData = { 
    type: "product", 
    name: `Scanned Product (${catName})`, 
    details: `Found Dimensions: ${objL} × ${objW} × ${objH} cm | Volume: ${Math.round(objL * objW * objH)} cm³` 
  };

  const prodEdgesGeo = new THREE.EdgesGeometry(prodGeo);
  const prodEdgesMat = new THREE.LineBasicMaterial({ color: 0x86efac, linewidth: 2 });
  const prodWire = new THREE.LineSegments(prodEdgesGeo, prodEdgesMat);
  productMesh.add(prodWire);

  packGroup.add(boxMesh);
  packGroup.add(paddingMesh);
  packGroup.add(productMesh);

  scene.add(packGroup);
}

/* ----------------------------------------------------
   3. RAYCASTER CLICK INSPECTOR
---------------------------------------------------- */
function onCanvasClick(event) {
  if (!renderer || !camera) return;

  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  const selectableObjects = [productMesh, paddingMesh, boxMesh].filter(Boolean);
  const intersects = raycaster.intersectObjects(selectableObjects, true);

  if (intersects.length > 0) {
    let hitObj = intersects[0].object;
    // Find top parent userData
    while (hitObj && !hitObj.userData.type && hitObj.parent) {
      hitObj = hitObj.parent;
    }

    if (hitObj && hitObj.userData.name) {
      document.getElementById("inspector-title").innerText = hitObj.userData.name;
      document.getElementById("inspector-desc").innerText = hitObj.userData.details;
    }
  }
}

/* ----------------------------------------------------
   4. DASHBOARD POLLING & INTERACTION
---------------------------------------------------- */
async function pollState() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();

    currentSource = data.camera_source;
    const navCam = document.getElementById("nav-camera-source");
    if (navCam) navCam.innerText = `Source: ${currentSource}`;

    currentStep = data.scan_state.step;
    updateControlBarUI(data.scan_state.step);

    if (data.scan_state.final_analysis) {
      renderAdvancedAnalytics(data.scan_state.final_analysis);
    }
  } catch (err) {
    console.warn("Poll error:", err);
  }
}

function updateControlBarUI(step) {
  const mainTitle = document.getElementById("status-main-title");
  const subTitle = document.getElementById("status-sub-title");
  const btnCapture = document.getElementById("btn-primary-capture");

  if (step === 1) {
    mainTitle.innerText = "Step 1: Place Product Flat";
    subTitle.innerText = "Position object flat under camera view and click Capture Image.";
    btnCapture.innerText = "Capture Image";
  } else if (step === 2) {
    mainTitle.innerText = "Step 2: Tilt Product on Side";
    subTitle.innerText = "Tilt object on side to measure height and click Capture Height Image.";
    btnCapture.innerText = "Capture Height Image";
  } else if (step === 3) {
    mainTitle.innerText = "Scan Complete";
    subTitle.innerText = "Advanced Packaging Analytics & 3D Visualization loaded below.";
    btnCapture.innerText = "View Advanced Analytics";
  }
}

async function handleMainCapture() {
  if (currentStep === 1) {
    const res = await fetch("/api/capture_step1", { method: "POST" });
    const data = await res.json();
    if (data.status === "success") {
      pollState();
    } else {
      alert(data.message || "Capture failed.");
    }
  } else if (currentStep === 2) {
    const res = await fetch("/api/capture_step2", { method: "POST" });
    const data = await res.json();
    if (data.status === "success") {
      renderAdvancedAnalytics(data.analysis);
      showAnalyticsPanel();
      pollState();
    } else {
      alert(data.message || "Analysis generation failed.");
    }
  } else if (currentStep === 3) {
    showAnalyticsPanel();
  }
}

async function resetScan() {
  await fetch("/api/reset_scan", { method: "POST" });
  hideAnalyticsPanel();
  pollState();
}

function showAnalyticsPanel() {
  const panel = document.getElementById("analytics-panel");
  if (panel) {
    panel.classList.add("active");
    panel.scrollIntoView({ behavior: "smooth" });
  }
}

function hideAnalyticsPanel() {
  const panel = document.getElementById("analytics-panel");
  if (panel) {
    panel.classList.remove("active");
  }
}

function renderAdvancedAnalytics(analysis) {
  if (!analysis) return;

  if (analysis.snapshot_base64) {
    document.getElementById("panel-snapshot").src = `data:image/jpeg;base64,${analysis.snapshot_base64}`;
  }

  const dims = analysis.dimensions;
  document.getElementById("panel-dims").innerText = dims.dimensions_str;
  document.getElementById("panel-vol").innerText = `${dims.volume_cm3.toLocaleString()} cm³`;

  const adv = analysis.advanced_metrics;
  const objM = adv.object_metrics;
  document.getElementById("panel-obj-weight").innerText = `${objM.estimated_weight_g} g (${objM.estimated_weight_kg} kg)`;
  document.getElementById("panel-category").innerText = `${analysis.category} (${Math.round(analysis.category_confidence * 100)}%)`;

  // Packaging Box Information
  const rules = analysis.packaging_rules;
  const opt = analysis.optimal_custom_box;
  const stk = analysis.best_stock_box;
  document.getElementById("panel-cardboard").innerText = rules.cardboard_type;
  document.getElementById("panel-opt-box").innerText = `${opt.dimensions_str} (${opt.volume_cm3} cm³)`;
  document.getElementById("panel-stock-box").innerText = stk ? `${stk.box_id} - ${stk.name} (${stk.dimensions_str})` : "OVERSIZED";

  // Space Utilization Dual Circular Ring Calculations (Radius 48 -> Circumference ~301.59)
  const circumference = 301.59;

  function updateRing(circleId, textId, subtextId, pct, defaultLabel) {
    const ringCircle = document.getElementById(circleId);
    const ringText = document.getElementById(textId);
    const ringSubtext = document.getElementById(subtextId);
    const offset = circumference - (pct / 100) * circumference;

    if (ringCircle) {
      ringCircle.style.strokeDasharray = `${circumference}`;
      ringCircle.style.strokeDashoffset = `${offset}`;

      // Color based on percentage: Red (<40%), Yellow (40-69%), Green (>=70%)
      if (pct < 40) {
        ringCircle.style.stroke = "#ef4444"; // Red
        ringCircle.style.filter = "drop-shadow(0 0 6px rgba(239, 68, 68, 0.6))";
        if (ringSubtext) ringSubtext.innerText = "Low Efficiency";
      } else if (pct < 70) {
        ringCircle.style.stroke = "#f59e0b"; // Yellow / Amber
        ringCircle.style.filter = "drop-shadow(0 0 6px rgba(245, 158, 11, 0.6))";
        if (ringSubtext) ringSubtext.innerText = "Moderate Fit";
      } else {
        ringCircle.style.stroke = "#22c55e"; // Green
        ringCircle.style.filter = "drop-shadow(0 0 6px rgba(34, 197, 94, 0.6))";
        if (ringSubtext) ringSubtext.innerText = defaultLabel || "Optimal Fit";
      }
    }
    if (ringText) {
      ringText.innerText = `${pct}%`;
    }
  }

  const optPct = Math.min(100, Math.max(0, Math.round(analysis.optimal_utilization_pct || (opt && opt.optimal_utilization_pct) || 92.5)));
  const stockPct = Math.min(100, Math.max(0, Math.round(analysis.stock_utilization_pct || analysis.space_utilization_pct || 0)));

  updateRing("panel-opt-ring-circle", "panel-opt-ring-text", "panel-opt-subtext", optPct, "Custom Tailored");
  updateRing("panel-stock-ring-circle", "panel-stock-ring-text", "panel-stock-subtext", stockPct, "Inventory Match");

  // Filling & Total Weight
  const fill = adv.filling_metrics;
  document.getElementById("panel-fill-name").innerText = fill.material_name;
  document.getElementById("panel-fill-vol").innerText = `${fill.void_volume_cm3.toLocaleString()} cm³`;
  document.getElementById("panel-fill-weight").innerText = `${fill.material_weight_g} g`;

  const tot = adv.total_package;
  document.getElementById("panel-total-weight").innerText = `${tot.total_weight_g} g (${tot.total_weight_kg} kg)`;

  document.getElementById("panel-handling-instruction").innerText = `INSTRUCTION: ${rules.handling_instructions}`;

  if (stk) {
    update3DScene(stk.length_cm, stk.width_cm, stk.height_cm, dims.length_cm, dims.width_cm, dims.height_cm, stk.box_id, analysis.category, fill.material_name);
  }
}

function promptCameraSource() {
  const newSrc = prompt("Enter Camera Source (IP URL e.g. http://192.168.1.50:8080/video or COM port):", currentSource);
  if (newSrc) {
    fetch("/api/change_camera", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: newSrc })
    });
  }
}
