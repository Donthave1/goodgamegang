(function($) {
  "use strict"; // Start of use strict

  // Smooth scrolling using jQuery easing
  $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function() {
    if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') && location.hostname == this.hostname) {
      var target = $(this.hash);
      target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
      if (target.length) {
        $('html, body').animate({
          scrollTop: (target.offset().top - 70)
        }, 1000, "easeInOutExpo");
        return false;
      }
    }
  });

  // Closes responsive menu when a scroll trigger link is clicked
  $('.js-scroll-trigger').click(function() {
    $('.navbar-collapse').collapse('hide');
  });

  // Activate scrollspy to add active class to navbar items on scroll
  $('body').scrollspy({
    target: '#mainNav',
    offset: 100
  });

  // Collapse Navbar
  var navbarCollapse = function() {
    if ($("#mainNav").offset().top > 100) {
      $("#mainNav").addClass("navbar-shrink");
    } else {
      $("#mainNav").removeClass("navbar-shrink");
    }
  };
  // Collapse now if page is not at top
  navbarCollapse();
  // Collapse the navbar when page is scrolled
  $(window).scroll(navbarCollapse);

})(jQuery); // End of use strict


// // function buildCharts() {
//   document.querySelector("#blackBox").addEventListener("click", function(e){
//     e.preventDefault();    //stop form from doing default submitting
//     let formData = new FormData()
//     const fileInput = document.querySelector('#fileSubmit');
//     formData.append('picture', fileInput.files[0]);

//     fetch('/upload', {
//       method: "POST",
//       body: formData
//     }).then(function(data) {
//       console.log(data)

//       // const svgArea = d3.select("#svgArea").select("svg");

//       // // clear svg is not empty
//       // if (!svgArea.empty()) {
//       //   svgArea.remove();
//       // }

//       // // Create an SVG wrapper, append an SVG group that will return result
//       // const svg = d3
//       //   .select("#svgArea")
//       //   .append("svg")
        


//     });
//   });
// // };