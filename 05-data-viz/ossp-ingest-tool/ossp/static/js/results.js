
/**
 * Automatically fetch and display all research question results on page load.
 */
window.onload = function () {
    getRQSelect({ value: "0" });
};
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
 * Gets research question answers based on what value is selected
 * @param {HTMLElement} selectObject 
 */
async function getRQSelect(selectObject) {
    const container = document.getElementById('errors-box');
    container.innerHTML = '';
    const params = new URLSearchParams();
    params.append('question', selectObject.value);
    try {
        const url = `/get/analysis?${params.toString()}`;
        const response = await fetch(url, { method: 'GET' });
        if (!response.ok) {
            response.json().then(error_message => {
                displayErrors(error_message);
            });
        } else {
            const rq_results = await response.json();
            renderResults(rq_results);
        }
    } catch (error) {
        console.log("Error occurred when fetching results for research question " + selectObject.value);
    }
};
/**
 * Renders json results onto page
 * @param {json} rq_results 
 */
function renderResults(rq_results) {
    const container = document.getElementById('rq-results-container');
    container.innerHTML = '';

    if (rq_results.length === 0) {
        container.innerHTML = 'No Results Available.';
        return;
    }

    // Group items by their research question title
    const groupedResults = {};
    rq_results.forEach(item => {
        const title = item["title"];
        if (!groupedResults[title]) {
            groupedResults[title] = [];
        }
        groupedResults[title].push(item);
    });

    // Iterate over each research question group
    Object.keys(groupedResults).forEach((title, groupIndex) => {
        // Add the research question text as a header
        const header = document.createElement("h4");
        header.classList.add("result-title");
        header.innerText = title;
        container.appendChild(header);

        // Separate non-html items and html items
        const htmlItems = [];
        const nonHtmlItems = [];
        groupedResults[title].forEach(item => {
            if (item["type"] === "html") {
                htmlItems.push(item);
            } else {
                nonHtmlItems.push(item);
            }
        });

        // Render non-html items (tables and images)
        nonHtmlItems.forEach(item => {
            const div = document.createElement("div");
            if (item["type"] === "table") {
                div.innerHTML = item["content"];
            } else if (item["type"] === "image") {
                div.classList.add("text-center", "mb-4");
                div.innerHTML = `<img src="${item["content"]}" class="img-fluid">`;
            } else {
                console.log("Invalid results type: " + item["type"]);
            }
            container.appendChild(div);
        });

        // Render html items using an accordion
        if (htmlItems.length > 0) {
            const accordionId = `accordion-${groupIndex}`;
            const accordionContainer = document.createElement("div");
            accordionContainer.classList.add("accordion", "mb-3");
            accordionContainer.id = accordionId;

            htmlItems.forEach((element, idx) => {
                const accordionItem = document.createElement("div");
                accordionItem.classList.add("accordion-item");

                const headingId = `heading-${groupIndex}-${idx}`;
                const collapseId = `collapse-${groupIndex}-${idx}`;
                const headerElem = document.createElement("h2");
                headerElem.classList.add("accordion-header");
                headerElem.id = headingId;
                const button = document.createElement("button");
                button.classList.add("accordion-button", "collapsed");
                button.type = "button";
                button.setAttribute("data-bs-toggle", "collapse");
                button.setAttribute("data-bs-target", `#${collapseId}`);
                button.setAttribute("aria-expanded", "false");
                button.setAttribute("aria-controls", collapseId);
                button.innerText = element["subtitle"];
                headerElem.appendChild(button);
                accordionItem.appendChild(headerElem);

                const accordionCollapse = document.createElement("div");
                accordionCollapse.id = collapseId;
                accordionCollapse.classList.add("accordion-collapse", "collapse");
                accordionCollapse.setAttribute("aria-labelledby", headingId);
                accordionCollapse.setAttribute("data-bs-parent", `#${accordionId}`);

                const accordionBody = document.createElement("div");
                accordionBody.classList.add("accordion-body");
                accordionBody.innerHTML = `
                            <iframe src="${element["content"]}" class="embed-responsive-item" style="width: 100%; height: 700px; border: none;" loading="lazy"></iframe>
                            <a href="${element["content"]}" target="_blank" class="btn btn-primary mt-3">View in New Tab</a>
                        `;
                accordionCollapse.appendChild(accordionBody);
                accordionItem.appendChild(accordionCollapse);
                accordionContainer.appendChild(accordionItem);
            });
            container.appendChild(accordionContainer);
        }
    });
}
