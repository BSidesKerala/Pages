document.addEventListener('DOMContentLoaded', function () {

    const targetDate = new Date("2025-02-08T09:00:00"); // Replace with your target date and time
    function updateCountdown() {
        const now = new Date();
        const timeDifference = targetDate - now;

        if (timeDifference < 0) {
            // If the countdown is complete, display a message
            document.getElementById("clock").innerHTML = "<h3>The event has started!</h3>";
            return;
        }

        // Calculate remaining days, hours, minutes, and seconds
        const days = Math.floor(timeDifference / (1000 * 60 * 60 * 24));
        const hours = Math.floor((timeDifference % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((timeDifference % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((timeDifference % (1000 * 60)) / 1000);

        // Update the HTML content
        document.querySelector("#clock .days span").textContent = days;
        document.querySelector("#clock .hours span").textContent = hours.toString().padStart(2, "0");
        document.querySelector("#clock .minutes span").textContent = minutes.toString().padStart(2, "0");
        document.querySelector("#clock .seconds span").textContent = seconds.toString().padStart(2, "0");
    }
    updateCountdown(); // Initial call to avoid delay
    setInterval(updateCountdown, 1000);
});
