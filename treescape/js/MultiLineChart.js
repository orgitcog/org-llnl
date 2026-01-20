ST.MultiLineChart = function() {

    var init_ = function () {
        const ctx = document.getElementById('myLineChart').getContext('2d');
        const myLineChart = new Chart(ctx, {
            type: 'line', // Line chart type
            data: {
                labels: ['January', 'February', 'March', 'April', 'May', 'June'], // X-axis labels
                datasets: [
                    {
                        label: 'Dataset 1', // Line 1
                        data: [12, 19, 3, 5, 2, 3],
                        borderColor: 'rgba(255, 99, 132, 1)', // Line color
                        backgroundColor: 'rgba(255, 99, 132, 0.2)', // Fill color (optional)
                        borderWidth: 2, // Line thickness
                        fill: true, // Area under the line
                    },
                    {
                        label: 'Dataset 2', // Line 2
                        data: [5, 15, 8, 12, 7, 10],
                        borderColor: 'rgba(54, 162, 235, 1)', // Line color
                        backgroundColor: 'rgba(54, 162, 235, 0.2)', // Fill color (optional)
                        borderWidth: 2,
                        fill: true,
                    },
                    {
                        label: 'Dataset 3', // Line 3
                        data: [8, 12, 6, 10, 4, 8],
                        borderColor: 'rgba(75, 192, 192, 1)', // Line color
                        backgroundColor: 'rgba(75, 192, 192, 0.2)', // Fill color (optional)
                        borderWidth: 2,
                        fill: true,
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true, // Show legend
                        position: 'top' // Legend position
                    },
                    tooltip: {
                        enabled: true // Show tooltips on hover
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Months' // X-axis title
                        }
                    },
                    y: {
                        beginAtZero: true, // Start y-axis at zero
                        title: {
                            display: true,
                            text: 'Values' // Y-axis title
                        }
                    }
                }
            }
        });
    }

    return {
        init: init_
    }
}();