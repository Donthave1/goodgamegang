// (function(){
//     var canvas = document.getElementById('canvas'), 
//         context = canvas.getContext('2d'),
//         video = document.getElementById('video'),
//         vendorUrl = window.URL || window.webkitURL;

//     navigator.getMedia = navigator.getUserMedia ||
//                         navigator.webkitGetUserMedia ||
//                         navigator.mozGetUserMedia ||
//                         navigator.msGetUserMedia;
//     console.log('before media');
//     navigator.getMedia({
//         video: true, 
//         audio: false
//     }, function(stream){
//         console.log('success media');
//         try {
//             this.srcObject = stream;
//           } catch (error) {
//             this.src = window.URL.createObjectURL(stream);
//           }
//         console.log(video.src);
//         video.play();
//     }, function(error) {
//         console.log('error media', error);
//         console.log('error media', navigator);
//         // An error occured 
//         // error.code
//     });

//     video.addEventListener('play', function(){
//         draw(this, context, 400, 300);
//     }, false);

//     function draw(video, context, width, height) {
//         context.drawImage(video, 0, 0, width, height);
//         setTimeout(draw, 10, video, context, width, height);
//     }

// })();


var video = document.querySelector("#video");

if (navigator.mediaDevices.getUserMedia) {       
    navigator.mediaDevices.getUserMedia({video: true})
  .then(function(stream) {
    video.srcObject = stream;
  })
  .catch(function(err0r) {
    console.log("Something went wrong!");
  });
}


const videoSelect = document.querySelector('select#videoSource');
navigator.mediaDevices.enumerateDevices()
  .then(gotDevices).then(getStream).catch(handleError);

videoSelect.onchange = getStream;

function gotDevices(deviceInfos) {
    for (let i = 0; i !== deviceInfos.length; ++i) {
      const deviceInfo = deviceInfos[i];
      const option = document.createElement('option');
      option.value = deviceInfo.deviceId;
     if (deviceInfo.kind === 'videoinput') {
        option.text = deviceInfo.label || 'camera ' +
          (videoSelect.length + 1);
        videoSelect.appendChild(option);
      } else {
        console.log('Found another kind of device: ', deviceInfo);
      }
    }
  }

