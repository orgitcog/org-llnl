
window.onload = loadScoringPage
/**
 * Downloads SBOM Scores from server
 */
function downloadScores() {
    console.log("Downloading scores excel.")
    // Define the URL for the download
    const url = '/download/scores';

    // Create a temporary anchor element to simulate the download
    const anchor = document.createElement('a');
    anchor.href = url;
    document.body.appendChild(anchor);
    anchor.click();

    // Clean up by removing the anchor element
    document.body.removeChild(anchor);
};
/**
 * Display Scoring Spinner
 */
function showSpinner() {
    document.getElementById('run-scoring-spinner').innerHTML = `
                <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Scoring...`;
};
/**
 * Hide Scoring Spinner
 */
function hideSpinner() {
    document.getElementById('run-scoring-spinner').innerHTML = `
                <i class="bi bi-play-fill me-2"></i>Rerun Scoring`;
};
/**
 * Displays described text on given elementId innerHTML
 * @param {string} elementid 
 * @param {string} successText 
 */
function updateSubmitButton(elementid, text) {
    let element = document.getElementById(elementid);
    element.innerHTML = text;
};
/**
 * Displays errors in error container
 * @param {string} error_message 
 */
function displayErrors(error_message) {
    const main_container = document.getElementById('main-container');
    // Create errors box
    main_container.innerHTML = '<div id="errors-box" class="section errors-box"></div>'

    const errors_container = document.getElementById('errors-box');
    const title = document.createElement('h3')
    title.classList.add("section-title")
    title.innerHTML = `Errors`
    errors_container.appendChild(title)

    const error = document.createElement('div')

    error.innerHTML = `<pre>${error_message['error']}</pre>`
    errors_container.appendChild(error)
};
/**
 * Loads scoring page by grabbing scores if they exist, or adding a message that scores dont exist
 */
async function loadScoringPage() {
    // Try to get scores
    let scores = await getScores()
    

    // If scores returned, render them
    if (Object.keys(scores).length > 0 && !("status" in scores)) {
        const errors_container = document.getElementById('errors-box'); // Remove errors if any are being show currently
        if (errors_container) { errors_container.remove() }
        renderScores(scores)
    } else if ("status" in scores) {
        if (scores["status"] == 400) {
            const scores_container = document.getElementById('scores-table-container');
            scores_container.innerHTML = "No SBOMs upload, please upload SBOMs before scoring."
        }
        else if(scores["status"] == 500) {
            const scores_container = document.getElementById('scores-table-container');
            scores_container.innerHTML = "No scores available. Please run scoring to view scores."
        } 
    } else {
        const scores_container = document.getElementById('scores-table-container');
        scores_container.innerHTML = "Error occured getting scores, please try again."
    }
};
/**
 * Sends POST request to server to run scoring 
 * @returns {string}
 */
async function runScoring() {
    try {
        const url = `/run/scoring`
        const response = await fetch(url, { method: 'POST' });
        if (!response.ok) {
            return null;
        } else {
            results = await response.json()
            console.log("Received scores successfully.")
            return results
        }
    } catch (error) {
        console.log("Error occured when fetching scores ")
        throw error
    }
};
/**
 * Retrives scores from server
 * @returns {json}
 */
async function getScores() {
    let scores;
    try {
        const url = `/get/scores`
        const response = await fetch(url, { method: 'GET' });
        if (!response.ok) {
            return {"status": response.status};
        } else {
            scores = await response.json()
            console.log("Received scores successfully.")
        }
    } catch (error) {
        console.log("Error occured when fetching scores", error)
    }
    return scores;
};
/**
 * Renders scores into html table
 * @param {json} scores 
 */
function renderScores(scores, renderSubscores) {
    const tableContainer = document.getElementById('scores-table-container');
    tableContainer.innerHTML = '';

    const table = document.createElement('table');
    table.id = "scores-table";
    table.classList.add("table", "table-responsive");

    const sboms = Object.entries(scores);
    const header = document.createElement('thead');
    var headerRow = document.createElement('tr');
    var rowSelect = document.createElement('th');
    rowSelect.innerHTML = `<input id="select-all" class="form-check-input" type="checkbox" checked value="">`;
    var rowFilename = document.createElement('th');
    rowFilename.innerHTML = "Filename";
    var rowAvgScore = document.createElement('th');
    rowAvgScore.innerHTML = "Average Score";

    headerRow.appendChild(rowSelect);
    headerRow.appendChild(rowFilename);
    headerRow.appendChild(rowAvgScore);

    if (renderSubscores) {
        for (let i = 0; i < sboms[0][1]['subscores'].length; i++) {
            var subscoreHeading = document.createElement('th');
            subscoreHeading.innerHTML = sboms[0][1]['subscores'][i]['feature']
            headerRow.appendChild(subscoreHeading);
        }
    };
    header.append(headerRow);
    table.appendChild(header);
    var body = document.createElement('tbody');

    for (const file in scores) {
        const newRow = createScoreRow(file, scores[file], renderSubscores);
        body.appendChild(newRow);
    }
    table.appendChild(body);
    tableContainer.appendChild(table);
    document.querySelectorAll('.table-options').forEach(item => {
        item.style.display = "flex";
    });


    document.getElementById("select-all").addEventListener('change', (event) => {
        updateSubmitButton("save-selections-btn", "Save Selections");
        if (event.currentTarget.checked) {
            document.querySelectorAll('.file-selector').forEach(item => {
                if (item.checkVisibility()) {
                    item.checked = true;
                }
            });
        } else {
            document.querySelectorAll('.file-selector').forEach(item => {
                if (item.checkVisibility()) {
                    item.checked = false;
                }
            });
        }
    });
    document.querySelectorAll(".file-selector").forEach(item => {
        item.addEventListener('change', function (e) {
            updateSubmitButton("save-selections-btn", "Save Selections");
        });
    });
};
/**
 * Creates a row in the scores table for given scores
 * @param {string} filename 
 * @param {json} scores 
 * @param {int} index 
 * @param {bool} renderSubscores 
 * @returns {HTMLElement} 
 */
function createScoreRow(filename, scores, renderSubscores) {
    var tableRow = document.createElement('tr');
    var rowSelect = document.createElement('td');
    if (scores["selected"] == true) {
        rowSelect.innerHTML = `<input id="select-${filename}" class="form-check-input file-selector" type="checkbox" checked value="">`;

    } else {
        rowSelect.innerHTML = `<input id="select-${filename}" class="form-check-input file-selector" type="checkbox" value="">`;
    }

    var rowFilename = document.createElement('td');
    rowFilename.innerHTML = `${filename}`;
    var rowScore = document.createElement('td');
    rowScore.classList.add("score-column");
    rowScore.innerHTML = `${scores["avg_score"].toFixed(2)}`;
    tableRow.appendChild(rowSelect);
    tableRow.appendChild(rowFilename);
    tableRow.appendChild(rowScore);

    if (renderSubscores) {
        for (let i = 0; i < scores['subscores'].length; i++) {
            var rowSubscore = document.createElement('td');
            rowSubscore.innerHTML = `${scores['subscores'][i]['score'].toFixed(2)}`;
            tableRow.appendChild(rowSubscore);
        }
    };
    return tableRow;
};

/**
 * Gets scores and renders if they exists, otherwise it will runs scoring function and render
 */
async function handleRunScoring() {
    showSpinner()
    const response = await runScoring()
    // If score running unsuccessful, return
    if (response === null) {
        hideSpinner()
        return
    }
    scores = await getScores()

    // If scores returned, render them
    if (Object.keys(scores).length > 0) {
        const scores_container = document.getElementById('scores-container');
        const errors_container = document.getElementById('errors-box'); // Remove errors if any are being show currently
        if (errors_container) { errors_container.remove() }
        renderScores(scores)
    } else {
        const scores_container = document.getElementById('scores-container');
        document.querySelectorAll('.table-options').forEach(item => {
            item.style.display = "none";
        });
        scores_container.innerHTML = "No Scores Available. Ensure SBOMs have been uploaded."
    }
    hideSpinner()
};

/**
 * Sends file selections as json to the /run/sbom_filtering endpoing
 */
function saveSbomSelections() {
    const sbomSelections = document.querySelectorAll(".file-selector");
    let selections = {};
    for (let i = 0; i < sbomSelections.length; i++) {
        var row = sbomSelections[i].closest("tr");
        if (row.checkVisibility()) {
            selections[sbomSelections[i].id.slice(7)] = sbomSelections[i].checked;
        } else {
            selections[sbomSelections[i].id.slice(7)] = false;
        }
    };
    const params = new URLSearchParams();
    params.append('min', document.getElementById('min-avg-score').value);
    fetch(`/run/sbom_filtering?${params.toString()}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json; charset=utf-8'
        },
        body: JSON.stringify(selections)
    })
        .then(response => {
            if (response.ok) {
                return response.json(); // Assuming the server responds with JSON
            } else {
                throw new Error('SBOM Filtering failed');
            }
        })
        .then(data => {
            // Handle success (e.g., show a success message)
            console.log('SBOM Filtering successful:', data);
            updateSubmitButton("save-selections-btn", "Selections Saved!");
        })
        .catch(error => {
            // Handle errors (e.g., show an error message)
            console.error('Error filtering sboms:', error);
            alert('Error filtering sboms, please try again.');
        });
};

/**
 * Hides and unchecks all score rows that dont fit min requirements
 */
function filter_on_min() {
    var scoreColumns = document.getElementsByClassName("score-column");
    for (var i = 0; i < scoreColumns.length; i++) {
        if (scoreColumns[i].innerHTML < document.getElementById('min-avg-score').value) {
            var row = scoreColumns[i].closest("tr");
            row.style = "display:none;"
            var select = row.getElementsByClassName("form-check-input");
            select[0].checked = false;
        } else {
            var row = scoreColumns[i].closest("tr");
            row.style = ""
            var select = row.getElementsByClassName("form-check-input");
            select[0].checked = true;
        }
    };
};
/**
 * Adds click event listener to run scoring button
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById('run-scoring-btn').addEventListener('click', handleRunScoring);

/**
 * Adds change event listener to subscoreToggle switch
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById("subscoreToggle").addEventListener('change', (event) => {
    if (event.currentTarget.checked) {
        renderScores(scores, true);

    } else {
        renderScores(scores, false);
    }
});

/**
 * Adds click event listener to save selections button
 * @type {HTMLElement}
 * @listens document#click
 */
document.getElementById("save-selections-btn").addEventListener('click', saveSbomSelections)

/**
 * Adds change event listener to min-avg-score input
 * @type {HTMLElement}
 * @listens document#change
 */
document.getElementById("min-avg-score").addEventListener('change', function (e) {
    updateSubmitButton("save-selections-btn", "Save Selections");
});

/**
 * Adds change event listener to min-avg-score input
 * @type {HTMLElement}
 * @listens document#change
 */
document.getElementById("min-avg-score").addEventListener('change', filter_on_min);




