//#region Helper Functions
/**
 * Dispalys Errors in Error Container
 * @param {string} error_message 
 */
function displayErrors(error_message) {
    const container = document.getElementById('errors-box');
    container.innerHTML = '';
    const title = document.createElement('h3');
    title.classList.add("section-title");
    title.innerHTML = `Errors`;
    container.appendChild(title);

    const error = document.createElement('div');

    error.innerHTML = `<pre>${error_message['error']}</pre>`;
    container.appendChild(error);
};
/**
 * Sends POST request to uri and performs action on success or failure.
 * @param {string} uri 
 * @param {Function} success_action 
 * @param {Function} failure_action 
 */
function runFunction(uri, success_action = null, failure_action = null) {
    fetch(uri, {
        method: 'POST'
    })
        .then(response => {
            if (response.ok) {
                if (success_action) { success_action() }
                return response.json();

            } else {
                return response.json().then(error_message => {
                    if (failure_action) { failure_action(error_message) }
                })

            }
        });
};
//#endregion
//#region Progress Bar Functions
/**
 * Displays progress bar within the given elementID
 * @param {string} elementid 
 */
function showProgressBar(elementid) {
    document.getElementById(elementid).innerHTML = `
                <div class="progress">
                    <div id="run-all-progress-bar" class="progress-bar progress-bar-striped progress-bar-animated bg-success" role="progressbar"
                        style="width: 0%" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                    </div>
                </div>
                `;
};
/**
 * Hides progress bar within the given elementID
 * @param {string} elementid 
 */
function hideProgressBar(elementid) {
    document.getElementById(elementid).innerHTML = `<span id="run-all-spinner"><i class="bi bi-play-circle me-2"></i>Run All Processes</span>`;
};
/**
 * Updates progress bar within the given progressBarId
 * @param {string} progressBarId 
 */
function updateProgressBar(progressBarId) {
    fetch("/run/progress", {
        method: 'GET'
    })
        .then(response => {
            if (response.ok) {
                return response.json()
            } else {
                throw new Error('Progress Check Failed');
            };
        }).then(data => {
            console.log(`Data recieved: ${data}`);
            let progressBar = document.getElementById(progressBarId);
            let percent = data["value"];
            const message = data["message"];
            console.log(`Process is ${percent}% finished.`);
            if (isNaN(percent)) {
                percent = 0;
            };
            progressBar.innerHTML = message;
            progressBar.ariaValueNow = percent;
            progressBar.style.width = `${percent}%`;

        })
        .catch(error => {
            console.error('Error getting progress status:', error);
        });
};
//#endregion
//#region Spinner Functions
/**
 * Displays spinner in the given elementId along with the loading text
 * @param {string} elementid 
 * @param {string} loadingtext 
 */
function showSpinner(elementid, loadingtext) {
    document.getElementById(elementid).innerHTML = `
                <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>${loadingtext}`;
};
/**
 * Hides spinner in the given elementId
 * @param {string} elementid 
 */
function hideSpinner(elementid) {
    let element = document.getElementById(elementid);
    element.innerHTML = `<i class="bi bi-play-circle me-2"></i>Run All Processes`;
};
//#endregion
//#region Submit Button Functions
/**
 * Updates text on given elementId innerHTML
 * @param {string} elementid 
 */
function updateBtnText(elementid, text) {
    let element = document.getElementById(elementid);
    element.innerHTML = text;
};

//#endregion
//#region EventListener Functions
/**
 * Displays file info for selected file
 * @param {Event} event - change invent on input file type
 */
function displayFileInfo(event) {
    const file = event.target.files[0];
    const fileInfoElement = this.closest('.card-body').querySelector('.file-info');

    if (file) {
        fileInfoElement.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
        const submit = this.closest('.card-body').querySelector('.submit-btn');
        updateBtnText(submit.id, "Submit")
    } else {
        fileInfoElement.textContent = '';
    };
};
/**
 * Downloads blank sbom zip file from /download/zip endpoint
 * @param {Event} event - change invent on input file type
 */
function downloadSBOMZip(event) {
    console.log("Downloading blank sbom zip.")
    // Define the URL for the download
    const url = '/download/zip';

    fetch(url, {
        method: 'GET',
    }).then(response => {
        if (response.ok) {
            console.log("Formatted Zip Exists.")
            return response.blob()
        } else {
            throw new Error('No formatted zip file exists.');
        }
    }).then(blob => {
        // Create a temporary anchor element to simulate the download
        const anchor = document.createElement('a');
        anchor.href = url;
        document.body.appendChild(anchor);
        anchor.click();

        // Clean up by removing the anchor element
        document.body.removeChild(anchor);
    }).catch(error => {
        // Handle errors (e.g., show an error message)
        console.error('Error fetching formatted zip:', error);
        alert('Error downloading formatted zip. Make sure asset list is uploaded and try again.');
    });
};
/**
 * Submits uploaded SBOM zip file to /submit/sboms endpoint
 * @param {Event} event 
 */
function submitSBOMZip(event) {
    // Get the file inputs
    const sbomsFileInput = document.getElementById('sboms-file');

    // Create a FormData object to hold the files
    const formData = new FormData();

    // Add the SBOMs file (if selected) to the FormData
    if (sbomsFileInput.files.length > 0) {
        formData.append('sboms', sbomsFileInput.files[0]);
    }

    console.log("Submit SBOMs Button Pressed");
    showSpinner('submit-sboms-btn', "")

    // Send the files to the server using fetch
    fetch('/submit/sboms', {
        method: 'POST',
        body: formData,
    })
        .then(response => {
            if (response.ok) {
                console.log("Success Response Recieved")
                updateBtnText("submit-sboms-btn", "Submitted!")
                const fileInfoElement = this.closest('.card-body').querySelector('.file-info');
                fileInfoElement.textContent = '';
                return response.json(); // Assuming the server responds with JSON
            } else {
                throw new Error('SBOMs upload failed');
            }
        })
        .then(data => {
            // Handle success (e.g., show a success message)
            console.log('Upload successful:', data);
            // alert('Files uploaded successfully!');
        })
        .catch(error => {
            // Handle errors (e.g., show an error message)
            console.error('Error uploading SBOMs:', error);
            alert('Error uploading SBOMs. Please try again.');
        });
};
/**
 * Submits organization info to /submit/org endpoint
 * @param {Event} event 
 */
function submitOrgInfo(event) {
    event.preventDefault()
    const formData = new FormData(this)

    console.log("Submit Button Pressed");

    // Send the files to the server using fetch
    fetch('/submit/org', {
        method: 'POST',
        body: formData,
    })
        .then(response => {
            if (response.ok) {
                updateBtnText('submit-org-btn', "Submitted!")
                return response.json(); // Assuming the server responds with JSON
            } else {
                throw new Error('Org submission failed');
            }
        })
        .then(data => {
            // Handle success (e.g., show a success message)
            console.log('Submission successful:', data);
            //alert('Org submission successful!');
        })
        .catch(error => {
            // Handle errors (e.g., show an error message)
            console.error('Error submitting org info:', error);
            alert('Error submitting org info. Please try again.');
        });
};
/**
 * Executes the run all processes
 * @param {Event} event 
 */
function runAllFunction(event) {
    const container = document.getElementById('errors-box');
    container.innerHTML = '';
    showProgressBar('run-all-btn');
    let intervalId = setInterval(() => { updateProgressBar('run-all-progress-bar') }, 5000)
    runFunction("/run", function () { hideProgressBar('run-all-btn'); clearInterval(intervalId); }, function (error_message) { displayErrors(error_message); hideProgressBar('run-all-btn'); clearInterval(intervalId); });
};
//#endregion
//#region Step 1: Upload Assets EventListener
/**
 * Adds a change listener to all file selector inputs
 * @type {HTMLElement}
 * @listens document#change
 */
document.querySelectorAll('.file-upload-btn input[type=file]').forEach(input => {
    input.addEventListener('change', displayFileInfo);
});
/**
 * Adds a click listener to download sbom zip button
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById("download-sbom-zip-btn").addEventListener("click", downloadSBOMZip);
document.getElementById('submit-assets-btn').addEventListener('click', function () {
    // Get the file inputs
    const assetsFileInput = document.getElementById('assets-file');

    // Create a FormData object to hold the files
    const formData = new FormData();

    // Add the Assets file (if selected) to the FormData
    if (assetsFileInput.files.length > 0) {
        formData.append('assets', assetsFileInput.files[0]);
    }
    console.log("Submit Assets Button Pressed");
    showSpinner('submit-assets-btn', "");
    // Send the files to the server using fetch
    fetch('/submit/assets', {
        method: 'POST',
        body: formData,
    })
        .then(response => {
            if (response.ok) {
                console.log("Success Response Recieved");
                updateBtnText("submit-assets-btn", "Submitted!");
                const fileInfoElement = this.closest('.card-body').querySelector('.file-info');
                fileInfoElement.textContent = '';
                return response.json(); // Assuming the server responds with JSON
            } else {
                throw new Error('Asset upload failed');
            }
        })
        .then(data => {
            // Handle success (e.g., show a success message)
            console.log('Upload successful:', data);
            // alert('Files uploaded successfully!');
        })
        .catch(error => {
            // Handle errors (e.g., show an error message)
            console.error('Error uploading assets:', error);
            alert('Error uploading assets. Please try again.');
        });
});
//#endregion
//#region Step 2: Upload SBOMs EventListener
/**
 * Adds a click listener to submit sboms button
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById('submit-sboms-btn').addEventListener('click', submitSBOMZip);
/**
 * Redirects window to scoring page on button click
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById('go-to-scoring-btn').addEventListener('click', function () {
    window.location.pathname = "/scores";
});
//#endregion
//#region Step 3: Input Organization Info EventListener
/**
 * Adds a submit listener to org-form submit
 * @type {HTMLElement}
 * @listens document#submit
 */
document.getElementById('org-form').addEventListener('submit', submitOrgInfo);
/**
 * Adds a change listener to org-form to reset submit button upon changes to form
 * @type {HTMLElement}
 * @listens document#change
 */
document.getElementById('org-form').addEventListener('change', function (e) {
    updateBtnText('submit-org-btn', 'Submit');
});
//#endregion
//#region Step 4: Input And Redact Info EventListener
/**
 * Redirects window to redaction page on button click
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById('run-redaction-btn').addEventListener('click', function () {
    window.location.pathname = "/redact";
});
/**
 * Executes the run all function upon button press
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById('run-all-btn').addEventListener('click', runAllFunction);
/**
 * Adds click listener to populate database btn
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById('populate-database-btn').addEventListener('click', function () {
    const container = document.getElementById('errors-box');
    container.innerHTML = '';
    runFunction("/run/populate_database", null, displayErrors)
});
/**
 * Adds click listener to run analysis btn
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById('run-analysis-btn').addEventListener('click', function () {
    const container = document.getElementById('errors-box');
    container.innerHTML = '';
    runFunction("/run/analysis", null, displayErrors)
});
/**
 * Adds click listener to refresh license btn
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById('refresh-license-btn').addEventListener('click', function () {
    const container = document.getElementById('errors-box');
    container.innerHTML = '';
    runFunction("/run/refresh_license_info", null, displayErrors)
});
/**
 * Adds click listener to infer licenses btn
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById('infer-license-btn').addEventListener('click', function () {
    const container = document.getElementById('errors-box');
    container.innerHTML = '';
    updateBtnText('infer-license-btn', 'Inferring In Progress!');
    runFunction("/run/infer_and_update_export", function () { updateBtnText('infer-license-btn', 'Inferring Complete!'); setTimeout(function () { updateBtnText('infer-license-btn', 'Infer Licenses'); }, 5000); }, displayErrors); // Connects to the infer_export endpoint
});
//#endregion