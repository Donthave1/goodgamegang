url = 


d3.json("/emo/").then({
    function(response){

        let emotion = []
        let 
        let time = []
        
        response.forEach(function(data){
            emotion.push(data.Emotion)
            time.push(data.Time)
        })

        var data = [{
            values: time,
            labels: emotion,
            type: 'pie'
          }];
          
          var layout = {
            height: 400,
            width: 500
          };
          
          Plotly.newPlot('myDiv', data, layout);



    }
  })