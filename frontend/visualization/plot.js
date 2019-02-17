function httpGetAsync(url, callback)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            callback(xmlHttp.responseText);
    }
    xmlHttp.open("GET", url, true); // true for asynchronous
    xmlHttp.send(null);
}

function sortByDate(array) {
	return array.sort(function(a,b){
	  return new Date(b.timestamp) - new Date(a.timestamp);
	});
}

function plotPowerTime(config) {

	var data = [];
	for (var i = 0; i < config.trace.length; i++) {
		data.push({
		  type: "scatter",
		  mode: "lines",
		  name: config.trace[i].name,
		  x: config.trace[i].x,
		  y: config.trace[i].y,
		  line: {color: config.trace[i].color}
		})
	}

	var layout = {
	  title: config.title,
	  xaxis: {
	    autorange: true,
	    range: [config.xRangeEnd, config.xRangeStart],
	    rangeselector: {buttons: [
	        {
	          count: 4,
	          label: '4h',
	          step: 'hour',
	          stepmode: 'backward'
	        },
	        {
	          count: 12,
	          label: '12h',
	          step: 'hour',
	          stepmode: 'backward'
	        },
	        {
	          count: 24,
	          label: '24h',
	          step: 'hour',
	          stepmode: 'backward'
	        },
	        {step: 'all'}
	      ]},
	    rangeslider: {range: [config.xRangeEnd, config.xRangeStart]},
	    type: 'date'
	  },
	  yaxis: {
	    autorange: true,
	    range: [config.yRangeEnd, config.yRangeStart],
	    type: 'linear'
	  }
	};

	Plotly.newPlot(config.elementId, data, layout);
}

// Consumption TODO: get data from API when CORS headers are working
userId = '1915Hall';
consumptionData = [{"user_id": "1915Hall", "power": 9.0, "timestamp": "2019-02-01T04:00:00Z"}, {"user_id": "1915Hall", "power": 9.0, "timestamp": "2019-02-01T05:00:00Z"}, {"user_id": "1915Hall", "power": 9.0, "timestamp": "2019-02-01T06:00:00Z"}, {"user_id": "1915Hall", "power": 9.0, "timestamp": "2019-02-01T07:00:00Z"}, {"user_id": "1915Hall", "power": 8.0, "timestamp": "2019-02-01T08:00:00Z"}, {"user_id": "1915Hall", "power": 8.0, "timestamp": "2019-02-01T09:00:00Z"}, {"user_id": "1915Hall", "power": 8.0, "timestamp": "2019-02-01T10:00:00Z"}, {"user_id": "1915Hall", "power": 8.0, "timestamp": "2019-02-01T11:00:00Z"}, {"user_id": "1915Hall", "power": 9.0, "timestamp": "2019-02-01T12:00:00Z"}, {"user_id": "1915Hall", "power": 6.0, "timestamp": "2019-02-01T13:00:00Z"}, {"user_id": "1915Hall", "power": 6.0, "timestamp": "2019-02-01T14:00:00Z"}, {"user_id": "1915Hall", "power": 7.0, "timestamp": "2019-02-01T15:00:00Z"}, {"user_id": "1915Hall", "power": 7.0, "timestamp": "2019-02-01T16:00:00Z"}, {"user_id": "1915Hall", "power": 7.0, "timestamp": "2019-02-01T17:00:00Z"}, {"user_id": "1915Hall", "power": 7.0, "timestamp": "2019-02-01T18:00:00Z"}, {"user_id": "1915Hall", "power": 11.0, "timestamp": "2019-02-01T19:00:00Z"}, {"user_id": "1915Hall", "power": 14.0, "timestamp": "2019-02-01T20:00:00Z"}, {"user_id": "1915Hall", "power": 9.0, "timestamp": "2019-02-01T21:00:00Z"}, {"user_id": "1915Hall", "power": 8.0, "timestamp": "2019-02-01T22:00:00Z"}, {"user_id": "1915Hall", "power": 10.0, "timestamp": "2019-02-01T23:00:00Z"}, {"user_id": "1915Hall", "power": 9.0, "timestamp": "2019-02-02T00:00:00Z"}, {"user_id": "1915Hall", "power": 11.0, "timestamp": "2019-02-02T01:00:00Z"}, {"user_id": "1915Hall", "power": 15.0, "timestamp": "2019-02-02T02:00:00Z"}, {"user_id": "1915Hall", "power": 12.0, "timestamp": "2019-02-02T03:00:00Z"}, {"user_id": "1915Hall", "power": 11.0, "timestamp": "2019-02-02T04:00:00Z"}, {"user_id": "1915Hall", "power": 12.0, "timestamp": "2019-02-02T05:00:00Z"}, {"user_id": "1915Hall", "power": 11.0, "timestamp": "2019-02-02T06:00:00Z"}, {"user_id": "1915Hall", "power": 11.0, "timestamp": "2019-02-02T07:00:00Z"}, {"user_id": "1915Hall", "power": 11.0, "timestamp": "2019-02-02T08:00:00Z"}, {"user_id": "1915Hall", "power": 12.0, "timestamp": "2019-02-02T09:00:00Z"}, {"user_id": "1915Hall", "power": 11.0, "timestamp": "2019-02-02T10:00:00Z"}, {"user_id": "1915Hall", "power": 11.0, "timestamp": "2019-02-02T11:00:00Z"}, {"user_id": "1915Hall", "power": 10.0, "timestamp": "2019-02-02T12:00:00Z"}];
// consumptionData = httpGetAsync('http://ec2-3-18-9-170.us-east-2.compute.amazonaws.com/get_past_consumption/' + userId, function(response) {
// 	console.log(response)
// });
consumptionData = sortByDate(consumptionData);
consumptionPower = consumptionData.map(function (obj) {
  return obj.power;
});
consumptionTime = consumptionData.map(function (obj) {
  return obj.timestamp;
});

productionTime = consumptionTime
productionPower = consumptionPower.map(x => x * 2);
console.log(productionPower)
productionPower = [16, 28, 15, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 15, 20, 26, 28, 22, 25, 23, 15, 1, 0, 0, 0]
console.log(productionPower)

var consumptionConfig = {
	elementId: 'consumption',
	title: '1927Hall - Consumption and Production',
	trace: [
		{
			name: 'Consumption',
			x: consumptionTime,
			y: consumptionPower,
			color: '#17BECF'
		},
		{
			name: 'Production',
			x: productionTime,
			y: productionPower,
			color: '#cf3517'
		}
	],
	xRangeStart: consumptionTime[0],
	xRangeEnd: consumptionTime[12],
	yRangeStart: consumptionPower[0],
	yRangeEnd: consumptionPower[12]
};

// 'Time (UTC),Boilers.1.NGas_dthr,Boilers.1.Oil_gpm,Boilers.2.NGas_dthr,Boilers.2.Oil_gpm,Chillers.i.Totals.ktons,Cogen.DB_NGas_dthr,Cogen.Turbine.NGas_dthr,Cogen.Turbine.output_kW,MicroTurbines.Total_kW,Plant.i.Campus_STMF_kpph,Plant.i.Total_STMF_kpph,Power.Charlton.Total_kW,Power.ElmSub.Total_kW,SMPL.SteamPress7041,WestWindsor.PV_kW,electricity.purchased_kW,electricity.total_kW,natural_gas.total_dthr'

// Production forecast TODO: get data from API when CORS headers are working
productionForecastData = [{"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-16T10:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-16T10:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-16T11:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-16T11:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-16T12:00:00Z"}, {"user_id": "solcast", "power": 12.5240656389438, "timestamp": "2019-02-16T12:30:00Z"}, {"user_id": "solcast", "power": 65.0615296676071, "timestamp": "2019-02-16T13:00:00Z"}, {"user_id": "solcast", "power": 234.010815051689, "timestamp": "2019-02-16T13:30:00Z"}, {"user_id": "solcast", "power": 344.865306909281, "timestamp": "2019-02-16T14:00:00Z"}, {"user_id": "solcast", "power": 430.899124748681, "timestamp": "2019-02-16T14:30:00Z"}, {"user_id": "solcast", "power": 508.66453256892, "timestamp": "2019-02-16T15:00:00Z"}, {"user_id": "solcast", "power": 579.949531585909, "timestamp": "2019-02-16T15:30:00Z"}, {"user_id": "solcast", "power": 643.458060017977, "timestamp": "2019-02-16T16:00:00Z"}, {"user_id": "solcast", "power": 695.234051447857, "timestamp": "2019-02-16T16:30:00Z"}, {"user_id": "solcast", "power": 730.737418726404, "timestamp": "2019-02-16T17:00:00Z"}, {"user_id": "solcast", "power": 750.665260109886, "timestamp": "2019-02-16T17:30:00Z"}, {"user_id": "solcast", "power": 739.945300635533, "timestamp": "2019-02-16T18:00:00Z"}, {"user_id": "solcast", "power": 711.942062602909, "timestamp": "2019-02-16T18:30:00Z"}, {"user_id": "solcast", "power": 656.524467759925, "timestamp": "2019-02-16T19:00:00Z"}, {"user_id": "solcast", "power": 573.915205289822, "timestamp": "2019-02-16T19:30:00Z"}, {"user_id": "solcast", "power": 484.586311413151, "timestamp": "2019-02-16T20:00:00Z"}, {"user_id": "solcast", "power": 386.919815798651, "timestamp": "2019-02-16T20:30:00Z"}, {"user_id": "solcast", "power": 297.896720880682, "timestamp": "2019-02-16T21:00:00Z"}, {"user_id": "solcast", "power": 217.033257946053, "timestamp": "2019-02-16T21:30:00Z"}, {"user_id": "solcast", "power": 129.172204715774, "timestamp": "2019-02-16T22:00:00Z"}, {"user_id": "solcast", "power": 24.6271733467918, "timestamp": "2019-02-16T22:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-16T23:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-16T23:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T00:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T00:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T01:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T01:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T02:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T02:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T03:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T03:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T04:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T04:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T05:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T05:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T06:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T06:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T07:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T07:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T08:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T08:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T09:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T09:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T10:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T10:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T11:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T11:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T12:00:00Z"}, {"user_id": "solcast", "power": 62.7026104795081, "timestamp": "2019-02-17T12:30:00Z"}, {"user_id": "solcast", "power": 187.43233452249, "timestamp": "2019-02-17T13:00:00Z"}, {"user_id": "solcast", "power": 307.472146677487, "timestamp": "2019-02-17T13:30:00Z"}, {"user_id": "solcast", "power": 418.41922446921, "timestamp": "2019-02-17T14:00:00Z"}, {"user_id": "solcast", "power": 518.976099562869, "timestamp": "2019-02-17T14:30:00Z"}, {"user_id": "solcast", "power": 601.161907146232, "timestamp": "2019-02-17T15:00:00Z"}, {"user_id": "solcast", "power": 658.780026122603, "timestamp": "2019-02-17T15:30:00Z"}, {"user_id": "solcast", "power": 697.912976862224, "timestamp": "2019-02-17T16:00:00Z"}, {"user_id": "solcast", "power": 712.113205240539, "timestamp": "2019-02-17T16:30:00Z"}, {"user_id": "solcast", "power": 682.910806296041, "timestamp": "2019-02-17T17:00:00Z"}, {"user_id": "solcast", "power": 631.573166934793, "timestamp": "2019-02-17T17:30:00Z"}, {"user_id": "solcast", "power": 555.966413887551, "timestamp": "2019-02-17T18:00:00Z"}, {"user_id": "solcast", "power": 471.796938635669, "timestamp": "2019-02-17T18:30:00Z"}, {"user_id": "solcast", "power": 380.138084966103, "timestamp": "2019-02-17T19:00:00Z"}, {"user_id": "solcast", "power": 298.558089750502, "timestamp": "2019-02-17T19:30:00Z"}, {"user_id": "solcast", "power": 224.859650427319, "timestamp": "2019-02-17T20:00:00Z"}, {"user_id": "solcast", "power": 161.509878030946, "timestamp": "2019-02-17T20:30:00Z"}, {"user_id": "solcast", "power": 104.109532386452, "timestamp": "2019-02-17T21:00:00Z"}, {"user_id": "solcast", "power": 54.5053328346401, "timestamp": "2019-02-17T21:30:00Z"}, {"user_id": "solcast", "power": 20.8999909528197, "timestamp": "2019-02-17T22:00:00Z"}, {"user_id": "solcast", "power": 4.54244458865648, "timestamp": "2019-02-17T22:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T23:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-17T23:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T00:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T00:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T01:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T01:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T02:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T02:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T03:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T03:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T04:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T04:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T05:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T05:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T06:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T06:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T07:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T07:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T08:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T08:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T09:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T09:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T10:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T10:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T11:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T11:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T12:00:00Z"}, {"user_id": "solcast", "power": 6.14020804746941, "timestamp": "2019-02-18T12:30:00Z"}, {"user_id": "solcast", "power": 19.9556761542756, "timestamp": "2019-02-18T13:00:00Z"}, {"user_id": "solcast", "power": 41.9410928207386, "timestamp": "2019-02-18T13:30:00Z"}, {"user_id": "solcast", "power": 60.5904694300973, "timestamp": "2019-02-18T14:00:00Z"}, {"user_id": "solcast", "power": 78.4051776658727, "timestamp": "2019-02-18T14:30:00Z"}, {"user_id": "solcast", "power": 92.1468336882878, "timestamp": "2019-02-18T15:00:00Z"}, {"user_id": "solcast", "power": 114.320291682705, "timestamp": "2019-02-18T15:30:00Z"}, {"user_id": "solcast", "power": 138.454730985575, "timestamp": "2019-02-18T16:00:00Z"}, {"user_id": "solcast", "power": 163.39578241818, "timestamp": "2019-02-18T16:30:00Z"}, {"user_id": "solcast", "power": 186.796900173337, "timestamp": "2019-02-18T17:00:00Z"}, {"user_id": "solcast", "power": 202.176472366985, "timestamp": "2019-02-18T17:30:00Z"}, {"user_id": "solcast", "power": 210.941243701924, "timestamp": "2019-02-18T18:00:00Z"}, {"user_id": "solcast", "power": 210.790334649816, "timestamp": "2019-02-18T18:30:00Z"}, {"user_id": "solcast", "power": 207.823139651854, "timestamp": "2019-02-18T19:00:00Z"}, {"user_id": "solcast", "power": 201.105456469125, "timestamp": "2019-02-18T19:30:00Z"}, {"user_id": "solcast", "power": 187.584386697739, "timestamp": "2019-02-18T20:00:00Z"}, {"user_id": "solcast", "power": 167.815658223487, "timestamp": "2019-02-18T20:30:00Z"}, {"user_id": "solcast", "power": 139.861165734427, "timestamp": "2019-02-18T21:00:00Z"}, {"user_id": "solcast", "power": 103.429887053446, "timestamp": "2019-02-18T21:30:00Z"}, {"user_id": "solcast", "power": 56.5277967950399, "timestamp": "2019-02-18T22:00:00Z"}, {"user_id": "solcast", "power": 12.2488864606764, "timestamp": "2019-02-18T22:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T23:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-18T23:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T00:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T00:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T01:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T01:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T02:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T02:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T03:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T03:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T04:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T04:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T05:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T05:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T06:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T06:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T07:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T07:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T08:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T08:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T09:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T09:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T10:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T10:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T11:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T11:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T12:00:00Z"}, {"user_id": "solcast", "power": 70.2347886222958, "timestamp": "2019-02-19T12:30:00Z"}, {"user_id": "solcast", "power": 199.527834554793, "timestamp": "2019-02-19T13:00:00Z"}, {"user_id": "solcast", "power": 322.416424174041, "timestamp": "2019-02-19T13:30:00Z"}, {"user_id": "solcast", "power": 433.235078198699, "timestamp": "2019-02-19T14:00:00Z"}, {"user_id": "solcast", "power": 534.166938905357, "timestamp": "2019-02-19T14:30:00Z"}, {"user_id": "solcast", "power": 614.058402224853, "timestamp": "2019-02-19T15:00:00Z"}, {"user_id": "solcast", "power": 686.571523999337, "timestamp": "2019-02-19T15:30:00Z"}, {"user_id": "solcast", "power": 735.556013795765, "timestamp": "2019-02-19T16:00:00Z"}, {"user_id": "solcast", "power": 772.58410099002, "timestamp": "2019-02-19T16:30:00Z"}, {"user_id": "solcast", "power": 782.536784763913, "timestamp": "2019-02-19T17:00:00Z"}, {"user_id": "solcast", "power": 778.785637800807, "timestamp": "2019-02-19T17:30:00Z"}, {"user_id": "solcast", "power": 753.331775326183, "timestamp": "2019-02-19T18:00:00Z"}, {"user_id": "solcast", "power": 716.924115694104, "timestamp": "2019-02-19T18:30:00Z"}, {"user_id": "solcast", "power": 653.259073177442, "timestamp": "2019-02-19T19:00:00Z"}, {"user_id": "solcast", "power": 579.893280858572, "timestamp": "2019-02-19T19:30:00Z"}, {"user_id": "solcast", "power": 495.535681964328, "timestamp": "2019-02-19T20:00:00Z"}, {"user_id": "solcast", "power": 395.926234086808, "timestamp": "2019-02-19T20:30:00Z"}, {"user_id": "solcast", "power": 286.445342909134, "timestamp": "2019-02-19T21:00:00Z"}, {"user_id": "solcast", "power": 178.225906077664, "timestamp": "2019-02-19T21:30:00Z"}, {"user_id": "solcast", "power": 82.5035116490587, "timestamp": "2019-02-19T22:00:00Z"}, {"user_id": "solcast", "power": 14.2096579542542, "timestamp": "2019-02-19T22:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T23:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-19T23:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T00:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T00:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T01:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T01:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T02:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T02:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T03:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T03:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T04:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T04:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T05:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T05:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T06:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T06:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T07:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T07:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T08:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T08:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T09:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T09:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T10:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T10:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T11:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T11:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T12:00:00Z"}, {"user_id": "solcast", "power": 5.86782003296272, "timestamp": "2019-02-20T12:30:00Z"}, {"user_id": "solcast", "power": 15.469707359629, "timestamp": "2019-02-20T13:00:00Z"}, {"user_id": "solcast", "power": 24.724117992022, "timestamp": "2019-02-20T13:30:00Z"}, {"user_id": "solcast", "power": 34.1085484931307, "timestamp": "2019-02-20T14:00:00Z"}, {"user_id": "solcast", "power": 42.8137197311465, "timestamp": "2019-02-20T14:30:00Z"}, {"user_id": "solcast", "power": 46.6759766374035, "timestamp": "2019-02-20T15:00:00Z"}, {"user_id": "solcast", "power": 48.6055570701281, "timestamp": "2019-02-20T15:30:00Z"}, {"user_id": "solcast", "power": 48.1501262859194, "timestamp": "2019-02-20T16:00:00Z"}, {"user_id": "solcast", "power": 49.1109794723807, "timestamp": "2019-02-20T16:30:00Z"}, {"user_id": "solcast", "power": 50.0715746554412, "timestamp": "2019-02-20T17:00:00Z"}, {"user_id": "solcast", "power": 51.0319118351012, "timestamp": "2019-02-20T17:30:00Z"}, {"user_id": "solcast", "power": 51.9919910113605, "timestamp": "2019-02-20T18:00:00Z"}, {"user_id": "solcast", "power": 51.9919910113605, "timestamp": "2019-02-20T18:30:00Z"}, {"user_id": "solcast", "power": 51.0319118351012, "timestamp": "2019-02-20T19:00:00Z"}, {"user_id": "solcast", "power": 46.2276459027951, "timestamp": "2019-02-20T19:30:00Z"}, {"user_id": "solcast", "power": 38.5274042342718, "timestamp": "2019-02-20T20:00:00Z"}, {"user_id": "solcast", "power": 28.8788818425597, "timestamp": "2019-02-20T20:30:00Z"}, {"user_id": "solcast", "power": 21.7893341416957, "timestamp": "2019-02-20T21:00:00Z"}, {"user_id": "solcast", "power": 15.2553068461412, "timestamp": "2019-02-20T21:30:00Z"}, {"user_id": "solcast", "power": 8.41672101856068, "timestamp": "2019-02-20T22:00:00Z"}, {"user_id": "solcast", "power": 2.63022531830021, "timestamp": "2019-02-20T22:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T23:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-20T23:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T00:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T00:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T01:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T01:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T02:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T02:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T03:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T03:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T04:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T04:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T05:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T05:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T06:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T06:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T07:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T07:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T08:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T08:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T09:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T09:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T10:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T10:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T11:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T11:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T12:00:00Z"}, {"user_id": "solcast", "power": 6.14020804746941, "timestamp": "2019-02-21T12:30:00Z"}, {"user_id": "solcast", "power": 17.6650622892196, "timestamp": "2019-02-21T13:00:00Z"}, {"user_id": "solcast", "power": 35.7962419159201, "timestamp": "2019-02-21T13:30:00Z"}, {"user_id": "solcast", "power": 63.2354442423501, "timestamp": "2019-02-21T14:00:00Z"}, {"user_id": "solcast", "power": 89.695942762565, "timestamp": "2019-02-21T14:30:00Z"}, {"user_id": "solcast", "power": 115.025697061024, "timestamp": "2019-02-21T15:00:00Z"}, {"user_id": "solcast", "power": 140.167083131807, "timestamp": "2019-02-21T15:30:00Z"}, {"user_id": "solcast", "power": 163.289950686965, "timestamp": "2019-02-21T16:00:00Z"}, {"user_id": "solcast", "power": 184.097844599708, "timestamp": "2019-02-21T16:30:00Z"}, {"user_id": "solcast", "power": 198.743926946724, "timestamp": "2019-02-21T17:00:00Z"}, {"user_id": "solcast", "power": 210.215341195632, "timestamp": "2019-02-21T17:30:00Z"}, {"user_id": "solcast", "power": 214.808835011148, "timestamp": "2019-02-21T18:00:00Z"}, {"user_id": "solcast", "power": 213.899438574441, "timestamp": "2019-02-21T18:30:00Z"}, {"user_id": "solcast", "power": 207.325795945696, "timestamp": "2019-02-21T19:00:00Z"}, {"user_id": "solcast", "power": 192.811023341368, "timestamp": "2019-02-21T19:30:00Z"}, {"user_id": "solcast", "power": 171.763335376996, "timestamp": "2019-02-21T20:00:00Z"}, {"user_id": "solcast", "power": 145.095254792879, "timestamp": "2019-02-21T20:30:00Z"}, {"user_id": "solcast", "power": 112.818618605472, "timestamp": "2019-02-21T21:00:00Z"}, {"user_id": "solcast", "power": 74.7335851858997, "timestamp": "2019-02-21T21:30:00Z"}, {"user_id": "solcast", "power": 34.3485868229937, "timestamp": "2019-02-21T22:00:00Z"}, {"user_id": "solcast", "power": 10.1386739380031, "timestamp": "2019-02-21T22:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T23:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-21T23:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T00:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T00:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T01:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T01:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T02:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T02:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T03:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T03:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T04:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T04:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T05:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T05:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T06:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T06:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T07:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T07:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T08:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T08:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T09:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T09:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T10:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T10:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T11:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T11:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T12:00:00Z"}, {"user_id": "solcast", "power": 21.723021971632, "timestamp": "2019-02-22T12:30:00Z"}, {"user_id": "solcast", "power": 85.1259677114172, "timestamp": "2019-02-22T13:00:00Z"}, {"user_id": "solcast", "power": 151.107111891256, "timestamp": "2019-02-22T13:30:00Z"}, {"user_id": "solcast", "power": 215.230447185586, "timestamp": "2019-02-22T14:00:00Z"}, {"user_id": "solcast", "power": 270.078631374394, "timestamp": "2019-02-22T14:30:00Z"}, {"user_id": "solcast", "power": 315.176147766293, "timestamp": "2019-02-22T15:00:00Z"}, {"user_id": "solcast", "power": 350.438907946707, "timestamp": "2019-02-22T15:30:00Z"}, {"user_id": "solcast", "power": 376.902800711249, "timestamp": "2019-02-22T16:00:00Z"}, {"user_id": "solcast", "power": 393.449175306641, "timestamp": "2019-02-22T16:30:00Z"}, {"user_id": "solcast", "power": 400.052219707736, "timestamp": "2019-02-22T17:00:00Z"}, {"user_id": "solcast", "power": 398.679038239126, "timestamp": "2019-02-22T17:30:00Z"}, {"user_id": "solcast", "power": 387.20329409516, "timestamp": "2019-02-22T18:00:00Z"}, {"user_id": "solcast", "power": 372.146465319835, "timestamp": "2019-02-22T18:30:00Z"}, {"user_id": "solcast", "power": 348.802836442144, "timestamp": "2019-02-22T19:00:00Z"}, {"user_id": "solcast", "power": 321.390033334667, "timestamp": "2019-02-22T19:30:00Z"}, {"user_id": "solcast", "power": 286.638370077594, "timestamp": "2019-02-22T20:00:00Z"}, {"user_id": "solcast", "power": 244.386207468502, "timestamp": "2019-02-22T20:30:00Z"}, {"user_id": "solcast", "power": 193.232200382655, "timestamp": "2019-02-22T21:00:00Z"}, {"user_id": "solcast", "power": 136.13942117229, "timestamp": "2019-02-22T21:30:00Z"}, {"user_id": "solcast", "power": 71.836551494866, "timestamp": "2019-02-22T22:00:00Z"}, {"user_id": "solcast", "power": 16.090207373749, "timestamp": "2019-02-22T22:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T23:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-22T23:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T00:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T00:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T01:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T01:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T02:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T02:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T03:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T03:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T04:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T04:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T05:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T05:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T06:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T06:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T07:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T07:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T08:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T08:30:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T09:00:00Z"}, {"user_id": "solcast", "power": 0.0, "timestamp": "2019-02-23T09:30:00Z"}]
productionForecastData = sortByDate(productionForecastData);
productionForecastPower = productionForecastData.map(function (obj) {
  return obj.power;
});
productionForecastTime = productionForecastData.map(function (obj) {
  return obj.timestamp;
});


consumptionForecastTime = productionForecastTime
consumptionForecastPower = productionForecastPower.map(x => x * 0.5 + 42);

var productionForecastConfig = {
	elementId: 'production_forecast',
	title: 'Campus - Production and Consumption Forecast',
	trace: [
		{
			name: 'Consumption forecast',
			x: consumptionForecastTime,
			y: consumptionForecastPower,
		},
		{
			name: 'Production forecast',
			x: productionForecastTime,
			y: productionForecastPower,
		}
	],
	xRangeStart: productionForecastTime[0],
	xRangeEnd: productionForecastTime[12],
	yRangeStart: productionForecastPower[0],
	yRangeEnd: productionForecastPower[12]
}

//
document.addEventListener("DOMContentLoaded", function() {
	plotPowerTime(consumptionConfig);
	plotPowerTime(productionForecastConfig);
});
