
d3.json("/emo").then(
    function(response){
        console.log(response)

        let emotion = []
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
            width: 500,
            paper_bgcolor:'rgba(0,0,0,0)',
            plot_bgcolor:'rgba(0,0,0,0)',
            font: {
                family: 'Courier New, monospace',
                size: 18,
                color: 'white'
              }
          };
        
          Plotly.newPlot('myDiv', data, layout);
    }
  )