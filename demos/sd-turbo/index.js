// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run sd-turbo with webnn in onnxruntime-web.
//

function log(i) { 
    console.log(i); 
    document.getElementById('status').innerText += `\n[${getDateTime()}] ${i}`;
}

/*
 * get configuration from url
*/
function getConfig() {
    const query = window.location.search.substring(1);
    var config = {
        model: location.href.includes("github.io") ? "https://huggingface.co/onnxruntime-web-temp/demo/resolve/main/sd-turbo" : "models",
        provider: "webnn",
        device: "gpu",
        threads: "1",
        images: "4",
    };
    let vars = query.split("&");
    for (var i = 0; i < vars.length; i++) {
        let pair = vars[i].split("=");
        if (pair[0] in config) {
            config[pair[0]] = decodeURIComponent(pair[1]);
        } else if (pair[0].length > 0) {
            throw new Error("unknown argument: " + pair[0]);
        }
    }
    config.threads = parseInt(config.threads);
    config.images = parseInt(config.images);
    return config;
}

/*
 * initialize latents with random noise
 */
function randn_latents(shape, noise_sigma) {
    function randn() {
        // Use the Box-Muller transform
        let u = Math.random();
        let v = Math.random();
        let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        return z;
    }
    let size = 1;
    shape.forEach(element => {
        size *= element;
    });

    let data = new Float32Array(size);
    // Loop over the shape dimensions
    for (let i = 0; i < size; i++) {
        data[i] = randn() * noise_sigma;
    }
    return data;
}

let textEncoderFetchProgress = 0;
let unetFetchProgress = 0;
let vaeDecoderFetchProgress = 0;
let textEncoderCompileProgress = 0;
let unetCompileProgress = 0;
let vaeDecoderCompileProgress = 0;

// Get model via Origin Private File System
async function getModelOPFS(name, url, updateModel) {
    const root = await navigator.storage.getDirectory();
    let fileHandle;

    async function updateFile() {
        const response = await fetch(url);
        const buffer = await readResponse(name, response);
        fileHandle = await root.getFileHandle(name, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(buffer);
        await writable.close();
        return buffer;
    }

    if (updateModel) {
        return await updateFile();
    }

    try {
        fileHandle = await root.getFileHandle(name);
        const blob = await fileHandle.getFile();
        let buffer = await blob.arrayBuffer();
        if (buffer) {
            if (name == 'text_encoder') {
                textEncoderFetchProgress = 20.00;
            } else if (name == 'unet') {
                unetFetchProgress = 50.00;
            } else if (name == 'vae_decoder') {
                vaeDecoderFetchProgress = 8.00;
            }

            progress = textEncoderFetchProgress + unetFetchProgress + vaeDecoderFetchProgress + textEncoderCompileProgress + unetCompileProgress + vaeDecoderCompileProgress;
            updateLoadWave(progress.toFixed(2));
            return buffer;
        }

    } catch (e) {
        return await updateFile();
    }
}

async function readResponse(name, response) {
    const contentLength = response.headers.get('Content-Length');
    let total = parseInt(contentLength ?? '0');
    let buffer = new Uint8Array(total);
    let loaded = 0;

    const reader = response.body.getReader();
    async function read() {
        const { done, value } = await reader.read();
        if (done) return;

        let newLoaded = loaded + value.length;
        fetchProgress = (newLoaded / contentLength) * 100;

        if (name == 'text_encoder') {
            textEncoderFetchProgress = 0.20 * fetchProgress;
        } else if (name == 'unet') {
            unetFetchProgress = 0.50 * fetchProgress;
        } else if (name == 'vae_decoder') {
            vaeDecoderFetchProgress = 0.08 * fetchProgress;
        }

        progress = textEncoderFetchProgress + unetFetchProgress + vaeDecoderFetchProgress + textEncoderCompileProgress + unetCompileProgress + vaeDecoderCompileProgress;

        updateLoadWave(progress.toFixed(2));

        if (newLoaded > total) {
            total = newLoaded;
            let newBuffer = new Uint8Array(total);
            newBuffer.set(buffer);
            buffer = newBuffer;
        }
        buffer.set(value, loaded);
        loaded = newLoaded;
        return read();
    }

    await read();
    return buffer;
}

/*
 * load models used in the pipeline
 */
async function load_models(models) {
    log("[Load] ONNX Runtime Execution Provider: " + config.provider);
    updateLoadWave(0.00);
    load.disabled = true;

    for (const [name, model] of Object.entries(models)) {
        let modelNameInLog = '';
        try {
            let start = performance.now();
            let modelUrl;
            if (name == 'text_encoder') {
                modelNameInLog = 'Text Encoder';
                modelUrl = `${config.model}/${name}/model_layernorm.onnx`;
            } else if (name == 'unet') {
                modelNameInLog = 'UNet';
                modelUrl = `${config.model}/${name}/model_layernorm.onnx`;
            } else if(name == 'vae_decoder') {
                modelNameInLog = 'VAE Decoder';
                modelUrl = `${config.model}/${name}/model.onnx`;
            }
            log(`[Load] Loading model ${modelNameInLog} · ${model.size}`);
            let modelBuffer = await getModelOPFS(name, modelUrl, false);
            let modelFetchTime = (performance.now() - start).toFixed(2);
            if (name == 'text_encoder') {
                textEncoderFetch.innerHTML = modelFetchTime;
            } else if (name == 'unet') {
                unetFetch.innerHTML = modelFetchTime;
            } else if(name == 'vae_decoder') {
                vaeFetch.innerHTML = modelFetchTime;
            }
            log(`[Load] ${modelNameInLog} loaded · ${modelFetchTime}ms`);
            log(`[Session Create] Beginning ${modelNameInLog}`);

            start = performance.now();
            const sess_opt = { ...opt, ...model.opt };
            console.log(sess_opt);
            models[name].sess = await ort.InferenceSession.create(modelBuffer, sess_opt);
            let createTime = (performance.now() - start).toFixed(2);

            if (name == 'text_encoder') {
                textEncoderCreate.innerHTML = createTime;
                textEncoderCompileProgress = 5;
                progress = textEncoderFetchProgress + unetFetchProgress + vaeDecoderFetchProgress + textEncoderCompileProgress + unetCompileProgress + vaeDecoderCompileProgress;
                updateLoadWave(progress.toFixed(2));
            } else if (name == 'unet') {
                unetCreate.innerHTML = createTime;
                unetCompileProgress = 15;
                progress = textEncoderFetchProgress + unetFetchProgress + vaeDecoderFetchProgress + textEncoderCompileProgress + unetCompileProgress + vaeDecoderCompileProgress;
                updateLoadWave(progress.toFixed(2));
            } else if(name == 'vae_decoder') {
                vaeCreate.innerHTML = createTime;
                vaeDecoderCompileProgress = 2;
                progress = textEncoderFetchProgress + unetFetchProgress + vaeDecoderFetchProgress + textEncoderCompileProgress + unetCompileProgress + vaeDecoderCompileProgress;
                updateLoadWave(progress.toFixed(2));
            }

            log(`[Session Create] Create ${modelNameInLog} completed · ${createTime}ms`);

        } catch (e) {
            log(`[Load] ${modelNameInLog} failed, ${e}`);
        }
    }

    updateLoadWave(100.00);
    log("[Session Create] Ready to generate images");
    let image_area = document.querySelectorAll('#image_area>div');
    image_area.forEach(i=> {
        i.setAttribute('class','frame ready');
    });
    buttons.setAttribute('class', 'button-group key loaded');
    generate.disabled = false;
    document.querySelector("#user-input").setAttribute('class', 'form-control enabled');
}

const config = getConfig();

const models = {
    "unet": {
        // original model from dw, then wm dump new one from local graph optimization.
        url: "unet/model_layernorm.onnx", 
        size: '1.61GB',
        opt: { graphOptimizationLevel: 'disabled' }, // avoid wasm heap issue (need Wasm memory 64)
    },
    "text_encoder": {
        // orignal model from gu, wm convert the output to fp16.
        url: "text_encoder/model_layernorm.onnx", 
        size: '649MB',
        opt: { graphOptimizationLevel: 'disabled' },
        // opt: { freeDimensionOverrides: { batch_size: 1, sequence_length: 77 } },
    },
    "vae_decoder": {
        // use gu's model has precision lose in webnn caused by instanceNorm op,
        // covert the model to run instanceNorm in fp32 (insert cast nodes).
        url: "vae_decoder/model.onnx",
        size: '94.5MB',
        // opt: { freeDimensionOverrides: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } }
        opt: { freeDimensionOverrides: { batch: 1, channels: 4, height: 64, width: 64 } }
    }
}

let progress = 0;
let inferenceProgress = 0;

let tokenizer;
let loading;
const sigma = 14.6146;
const gamma = 0;
const vae_scaling_factor = 0.18215;

const opt = {
    executionProviders: [config.provider],
    enableMemPattern: false,
    enableCpuMemArena: false,
    extra: {
        session: {
            disable_prepacking: "1",
            use_device_allocator_for_initializers: "1",
            use_ort_model_bytes_directly: "1",
            use_ort_model_bytes_for_initializers: "1"
        }
    },
};

/*
 * scale the latents
*/
function scale_model_inputs(t) {
    const d_i = t.data;
    const d_o = new Float32Array(d_i.length);

    const divi = (sigma ** 2 + 1) ** 0.5;
    for (let i = 0; i < d_i.length; i++) {
        d_o[i] = d_i[i] / divi;
    }
    return new ort.Tensor(d_o, t.dims);
}

/*
 * Poor mens EulerA step
 * Since this example is just sd-turbo, implement the absolute minimum needed to create an image
 * Maybe next step is to support all sd flavors and create a small helper model in onnx can deal
 * much more efficient with latents.
 */
function step(model_output, sample) {
    const d_o = new Float32Array(model_output.data.length);
    const prev_sample = new ort.Tensor(d_o, model_output.dims);
    const sigma_hat = sigma * (gamma + 1);

    for (let i = 0; i < model_output.data.length; i++) {
        const pred_original_sample = sample.data[i] - sigma_hat * model_output.data[i];
        const derivative = (sample.data[i] - pred_original_sample) / sigma_hat;
        const dt = 0 - sigma_hat;
        d_o[i] = (sample.data[i] + derivative * dt) / vae_scaling_factor;
    }
    return prev_sample;
}

/**
 * draw an image from tensor
 * @param {ort.Tensor} t
 * @param {number} image_nr
*/
function draw_image(t, image_nr) {
    let pix = t.data;
    for (var i = 0; i < pix.length; i++) {
        let x = pix[i];
        x = x / 2 + 0.5
        if (x < 0.) x = 0.;
        if (x > 1.) x = 1.;
        pix[i] = x;
    }
    const imageData = t.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
    const canvas = document.getElementById(`img_canvas_${image_nr}`);
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext('2d').putImageData(imageData, 0, 0);
    const div = document.getElementById(`img_div_${image_nr}`);
    div.style.opacity = 1.
}

async function generate_image() {
    const img_divs = [img_div_0, img_div_1, img_div_2, img_div_3];
    img_divs.forEach(div => div.setAttribute('class', 'frame'));

    try {
        textEncoderRun1.innerHTML = '';
        textEncoderRun2.innerHTML = '';
        textEncoderRun3.innerHTML = '';
        textEncoderRun4.innerHTML = '';
        unetRun1.innerHTML = '';
        unetRun2.innerHTML = '';
        unetRun3.innerHTML = '';
        unetRun4.innerHTML = '';
        runTotal1.innerHTML = '';
        runTotal2.innerHTML = '';
        runTotal3.innerHTML = '';
        runTotal4.innerHTML = '';

       // document.querySelector(`#data1`).innerHTML = '... ms';
       // document.querySelector(`#data2`).innerHTML = '... ms';
       // document.querySelector(`#data3`).innerHTML = '... ms';
       // document.querySelector(`#data4`).innerHTML = '... ms';

        log(`[Session Run] Beginning`);

        await loading;

        for (let j = 0; j < config.images; j++) {
            const div = document.getElementById(`img_div_${j}`);
            div.style.opacity = 0.5
        }

        const prompt = document.querySelector("#user-input");
        const { input_ids } = await tokenizer(prompt.value, { padding: true, max_length: 77, truncation: true, return_tensor: false });

        // text_encoder
        let start = performance.now();
        const { last_hidden_state } = await models.text_encoder.sess.run(
            { "input_ids": new ort.Tensor("int32", input_ids, [1, input_ids.length]) });
        let sessionRunTimeTextEncode = (performance.now() - start).toFixed(2);
        textEncoderRun1.innerHTML = sessionRunTimeTextEncode;
        textEncoderRun2.innerHTML = sessionRunTimeTextEncode;
        textEncoderRun3.innerHTML = sessionRunTimeTextEncode;
        textEncoderRun4.innerHTML = sessionRunTimeTextEncode;
        log(`[Session Run] Text encode execution time: ${sessionRunTimeTextEncode}ms`);

        for (let j = 0; j < config.images; j++) {
            document.getElementById(`img_div_${j}`).setAttribute('class','frame inferncing');
            let startTotal = performance.now();
            const latent_shape = [1, 4, 64, 64];
            let latent = new ort.Tensor(randn_latents(latent_shape, sigma), latent_shape);
            const latent_model_input = scale_model_inputs(latent);

            // unet
            start = performance.now();
            let feed = {
                "sample": new ort.Tensor("float16", convertToUint16Array(latent_model_input.data), latent_model_input.dims),
                "timestep": new ort.Tensor("float16", new Uint16Array([toHalf(999)]), [1]),
                "encoder_hidden_states": last_hidden_state,
            };
            let { out_sample } = await models.unet.sess.run(feed);
            let unetRunTime = (performance.now() - start).toFixed(2);
            document.getElementById(`unetRun${j+1}`).innerHTML = unetRunTime;
            log(`[Session Run][Image ${j+1}] UNet execution time: ${unetRunTime}ms`);

            // scheduler
            const new_latents = step(new ort.Tensor("float32", convertToFloat32Array(out_sample.data), out_sample.dims), latent);

            // vae_decoder
            start = performance.now();
            const { sample } = await models.vae_decoder.sess.run({ "latent_sample": new_latents });
            let vaeRunTime = (performance.now() - start).toFixed(2);
            document.getElementById(`vaeRun${j+1}`).innerHTML = vaeRunTime;
            log(`[Session Run][Image ${j+1}] VAE decode execution time: ${vaeRunTime}ms`);
            document.getElementById(`img_div_${j}`).setAttribute('class','frame done');
            draw_image(sample, j);
            let totalRunTime = (performance.now() + Number(sessionRunTimeTextEncode) - startTotal ).toFixed(2);
            log(`[Total] Image ${j+1} execution time: ${totalRunTime}ms`);
            document.getElementById(`runTotal${j+1}`).innerHTML = totalRunTime;
            document.querySelector(`#data${j+1}`).innerHTML = totalRunTime + 'ms';
            document.querySelector(`#data${j+1}`).setAttribute('class', 'show');
        }
        // this is a gpu-buffer we own, so we need to dispose it
        last_hidden_state.dispose();
        log("[Info] Images generation completed");
    } catch (e) {
        log('[Error] ' + e);
    }
}

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter()
        return adapter.features.has('shader-f16')
    } catch (e) {
        return false
    }
}

// ref: http://stackoverflow.com/questions/32633585/how-do-you-convert-to-half-floats-in-javascript
const toHalf = (function () {

    var floatView = new Float32Array(1);
    var int32View = new Int32Array(floatView.buffer);

    /* This method is faster than the OpenEXR implementation (very often
     * used, eg. in Ogre), with the additional benefit of rounding, inspired
     * by James Tursa?s half-precision code. */
    return function toHalf(val) {

        floatView[0] = val;
        var x = int32View[0];

        var bits = (x >> 16) & 0x8000; /* Get the sign */
        var m = (x >> 12) & 0x07ff; /* Keep one extra bit for rounding */
        var e = (x >> 23) & 0xff; /* Using int is faster here */

        /* If zero, or denormal, or exponent underflows too much for a denormal
         * half, return signed zero. */
        if (e < 103) {
            return bits;
        }

        /* If NaN, return NaN. If Inf or exponent overflow, return Inf. */
        if (e > 142) {
            bits |= 0x7c00;
            /* If exponent was 0xff and one mantissa bit was set, it means NaN,
             * not Inf, so make sure we set one mantissa bit too. */
            bits |= ((e == 255) ? 0 : 1) && (x & 0x007fffff);
            return bits;
        }

        /* If exponent underflows but not too much, return a denormal */
        if (e < 113) {
            m |= 0x0800;
            /* Extra rounding may overflow and set mantissa to 0 and exponent
             * to 1, which is OK. */
            bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
            return bits;
        }

        bits |= ((e - 112) << 10) | (m >> 1);
        /* Extra rounding. An overflow will set mantissa to 0 and increment
         * the exponent, which is OK. */
        bits += m & 1;
        return bits;
    };

})();

// This function converts a Float16 stored as the bits of a Uint16 into a Javascript Number.
// Adapted from: https://gist.github.com/martinkallman/5049614
// input is a Uint16 (eg, new Uint16Array([value])[0])

function float16ToNumber(input) {
    // Create a 32 bit DataView to store the input
    const arr = new ArrayBuffer(4);
    const dv = new DataView(arr);

    // Set the Float16 into the last 16 bits of the dataview
    // So our dataView is [00xx]
    dv.setUint16(2, input, false);

    // Get all 32 bits as a 32 bit integer
    // (JS bitwise operations are performed on 32 bit signed integers)
    const asInt32 = dv.getInt32(0, false);

    // All bits aside from the sign
    let rest = asInt32 & 0x7fff;
    // Sign bit
    let sign = asInt32 & 0x8000;
    // Exponent bits
    const exponent = asInt32 & 0x7c00;

    // Shift the non-sign bits into place for a 32 bit Float
    rest <<= 13;
    // Shift the sign bit into place for a 32 bit Float
    sign <<= 16;

    // Adjust bias
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding
    rest += 0x38000000;
    // Denormals-as-zero
    rest = (exponent === 0 ? 0 : rest);
    // Re-insert sign bit
    rest |= sign;

    // Set the adjusted float32 (stored as int32) back into the dataview
    dv.setInt32(0, rest, false);

    // Get it back out as a float32 (which js will convert to a Number)
    const asFloat32 = dv.getFloat32(0, false);

    return asFloat32;
}

// convert Uint16Array to Float32Array
function convertToFloat32Array(fp16_array) {
    const fp32_array = new Float32Array(fp16_array.length);
    for (let i = 0; i < fp32_array.length; i++) {
        fp32_array[i] = float16ToNumber(fp16_array[i]);
    }
    return fp32_array;
}

// convert Float32Array to Uint16Array
function convertToUint16Array(fp32_array) {
    const fp16_array = new Uint16Array(fp32_array.length);
    for (let i = 0; i < fp16_array.length; i++) {
        fp16_array[i] = toHalf(fp32_array[i]);
    }
    return fp16_array;
}

const padNumber = (num, fill) => {
    let len = ('' + num).length;
    return Array(fill > len ? fill - len + 1 || 0 : 0).join(0) + num;
};

const getDateTime = () => {
    let date = new Date(),
        m = padNumber(date.getMonth() + 1, 2),
        d = padNumber(date.getDate(), 2),
        hour = padNumber(date.getHours(), 2),
        min = padNumber(date.getMinutes(), 2),
        sec = padNumber(date.getSeconds(), 2);
    return `${m}/${d} ${hour}:${min}:${sec}`;
};

const getOrtDevVersion = async () => {
    const response = await fetch('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/');
    const htmlString = await response.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlString, 'text/html');
    let selectElement = doc.querySelector('.path li');
    selectElement = doc.querySelector('select.versions.select-css');
    const options = Array.from(selectElement.querySelectorAll('option')).map(
        (option) => option.value
    );
    return options[0].replace('onnxruntime-web@', '');
};

const checkWebNN = async () => {
    let status = document.querySelector('#webnnstatus');
    let info = document.querySelector('#info');
    let webnnStatus = await webNnStatus();

    if (webnnStatus.webnn) {
        status.setAttribute('class', 'green');
        info.innerHTML = 'WebNN supported';
    } else {
        if (webnnStatus.error) {
            status.setAttribute('class', 'red');
            info.innerHTML = 'WebNN not supported: ' + webnnStatus.error;
        } else {
            status.setAttribute('class', 'red');
            info.innerHTML = 'WebNN not supported';
        }
    }

    if (getQueryValue('provider') && getQueryValue('provider').toLowerCase().indexOf('webgpu') > -1) {
        status.innerHTML = '';
    }
};

const webNnStatus = async () => {
    let result = {};
    try {
        const context = await navigator.ml.createContext();
        if (context) {
            try {
                const builder = new MLGraphBuilder(context);
                if (builder) {
                    result.webnn = true;
                    return result;
                } else {
                    result.webnn = false;
                    return result;
                }
            } catch (e) {
                result.webnn = false;
                result.error = e.message;
                return result;
            }
        } else {
            result.webnn = false;
            return result;
        }
    } catch (ex) {
        result.webnn = false;
        result.error = ex.message;
        return result;
    }
};

const setupORT = async () => {
    const ortversion = document.querySelector('#ortversion');
    removeElement('onnxruntime-web');
    let ortVersion = await getOrtDevVersion();
    let ortLink = '';
    if (ortVersion && ortVersion.length > 4) {
        await loadScript('onnxruntime-web', `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/ort.all.min.js`);
        ortLink = `https://www.npmjs.com/package/onnxruntime-web/v/${ortVersion}`
        ortversion.innerHTML = `ONNX Runtime Web: <a href="${ortLink}">${ortVersion}</a>`;
    } else {
        await loadScript('onnxruntime-web', './dist/ort.all.min.js');
        ortversion.innerHTML = `ONNX Runtime Web: Test version`;
    }
}

const loadScript = async (id, url) => {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.onload = resolve;
        script.onerror = reject;
        script.id = id;
        script.src = url;
        if (url.startsWith('http')) {
            script.crossOrigin = 'anonymous';
        }
        document.body.append(script);
    })
}

const removeElement = async (id) => {
    let el = document.querySelector(id);
    if (el) {
        el.parentNode.removeChild(el);
    }
}

const getQueryValue = (name) => {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

let textEncoderFetch = null;
let textEncoderCreate = null;
let textEncoderRun1 = null;
let textEncoderRun2 = null;
let textEncoderRun3 = null;
let textEncoderRun4 = null;
let unetFetch = null;
let unetCreate = null;
let vaeFetch = null;
let vaeCreate = null;
let unetRun1 = null;
let unetRun2 = null;
let unetRun3 = null;
let unetRun4 = null;
let runTotal1 = null;
let runTotal2 = null;
let runTotal3 = null;
let runTotal4 = null;
let generate = null;
let load = null;
let buttons = null;
let loadwave = null;
let loadwaveData = null; 

const updateLoadWave = (value) => {
    loadwave = document.querySelectorAll('.loadwave');
    loadwaveData = document.querySelectorAll('.loadwave-data strong');

    if(loadwave && loadwaveData) {
        loadwave.forEach(l => {
            l.style.setProperty(`--loadwave-value`, value);
        })
        loadwaveData.forEach(data => {
            data.innerHTML = value;
        });

        if(value === 100) {
            loadwave.forEach(l => {
                l.dataset.value = value;
            })
        }
    }
}

const ui = async () => {
    await setupORT();

    const title = document.querySelector('#title');
    if (getQueryValue('provider') && getQueryValue('provider').toLowerCase().indexOf('webgpu') > -1) {
        title.innerHTML = 'WebGPU';
    }
    await checkWebNN();

    // const img_div_ids = ["#img_div_0", "#img_div_1", "#img_div_2", "#img_div_3"];
    // [img_div_0, img_div_1, img_div_2, img_div_3] = img_div_ids.map(id => document.querySelector(id));

    const elementIds = [
        "#textEncoderFetch", 
        "#textEncoderCreate", 
        "#textEncoderRun1", 
        "#textEncoderRun2", 
        "#textEncoderRun3", 
        "#textEncoderRun4", 
        "#unetRun1",
        "#unetRun2",
        "#unetRun3",
        "#unetRun4",
        "#runTotal1",
        "#runTotal2",
        "#runTotal3",
        "#runTotal4",
        "#unetFetch", 
        "#unetCreate", 
        "#vaeFetch", 
        "#vaeCreate"
    ];

    [
        textEncoderFetch, 
        textEncoderCreate, 
        textEncoderRun1, 
        textEncoderRun2, 
        textEncoderRun3, 
        textEncoderRun4, 
        unetRun1,
        unetRun2,
        unetRun3,
        unetRun4,
        runTotal1,
        runTotal2,
        runTotal3,
        runTotal4,
        unetFetch, 
        unetCreate, 
        vaeFetch, 
        vaeCreate
    ] = elementIds.map(id => document.querySelector(id));

    switch (config.provider) {
        case "webgpu":
            if (!("gpu" in navigator)) {
                throw new Error("webgpu is NOT supported");
            }
            opt.preferredOutputLocation = { last_hidden_state: "gpu-buffer" };
            break;
        case "webnn":
            let webnnStatus = await webNnStatus();
            if (webnnStatus.webnn) {
                opt.executionProviders = [{
                    name: "webnn",
                    deviceType: config.device,
                    powerPreference: 'default'
                }];
            }
            break;
    }
    
    const prompt = document.querySelector("#user-input");
    
    load = document.querySelector("#load");
    load.disabled = false;
    generate = document.querySelector("#generate");
    generate.disabled = true;
    buttons = document.querySelector('#buttons');
    prompt.value = "a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo";
    // Event listener for Ctrl + Enter or CMD + Enter
    prompt.addEventListener('keydown', function (e) {
        if (e.ctrlKey && e.key === 'Enter') {
            generate_image();
        }
    });
    generate.addEventListener('click', function (e) {
        generate_image()
    });

    const load_model_ui = () => {
        loading = load_models(models);
        const img_divs = [img_div_0, img_div_1, img_div_2, img_div_3];
        img_divs.forEach(div => div.setAttribute('class', 'frame loadwave'));
        buttons.setAttribute('class', 'button-group key loading');
    }

    load.addEventListener('click', ()=> {
        if (config.provider === 'webgpu') {
            hasFp16().then((fp16) => {
                if (fp16) {
                    load_model_ui();
                } else {
                    log(`[Error] Your GPU or Browser doesn't support webgpu/f16`);
                }
            });
        } else {
            load_model_ui();
        } 
    })

    // ort.env.wasm.wasmPaths = 'dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    let path = '';
    if (location.href.toLowerCase().indexOf('github.io') > -1 
    || location.href.toLowerCase().indexOf('huggingface.co') > -1
    || location.href.toLowerCase().indexOf('vercel.app') > -1
    || location.href.toLowerCase().indexOf('onnxruntime-web-demo') > -1) {
        path = 'onnxruntime-web-temp/demo/resolve/main/sd-turbo/tokenizer';        
    } else {
        path = '../../demos/sd-turbo/models/tokenizer'
    }

    tokenizer = await AutoTokenizer.from_pretrained(path);
    tokenizer.pad_token_id = 0;
};

document.addEventListener('DOMContentLoaded', ui, false);