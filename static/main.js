 // window.onload=function(){
//     // document.querySelector('h2').style.color='red';

// }

// class VoiceRecorder{
//     constructor(){
//         if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
//             console.log("Get user media suported");

//         }else{
//             console.log("Get user media not suported");

//         }
//         this.mediaRecorder
//         this.stream
//         this.chunks =[]
//         this.isRecording = false

//         this.recorderRef=document.querySelector("#recorder");
//         this.playerRef=document.querySelector("#player");
//         this.startRef=document.querySelector("#start");
//         this.stopRef=document.querySelector("#stop");

//         this.startRef.onclick=this.startRecording.bind(this);
//         this.stopRef.onclick=this.stopRecording.bind(this);

//         this.constraints ={
//             audio:true,
//             video:false
//         }

//     }
//     //handle sucess
//     handleSucess(stream){
//         this.stream=stream
//         this.stream.onclick=()=>{
//             console.log("stream ended");
//         }
//         this.recorderRef.srcObject=this.stream
//         this.mediaRecorder=new MediaRecorder(this.stream)
//         this.mediaRecorder.ondataavailable=this.onMediaRecorderDataAvailable.bind(this);
//         this.mediaRecorder.onstop=this.onMediaRecorderStop.bind(this);
//         this.recorderRef.play();
//         this.mediaRecorder.start();
//     }
//     onMediaRecorderDataAvailable(e){this.chunks.push(e.data)}
//     onMediaRecorderStop(e){
//         const blob = new Blob(this.chunks,{'type':'audio/wav; codesc=opus'});
//         const audioUrl= window.URL.createObjectURL(blob);
//         this.playerRef.src=audioUrl;
//         this.chunks=[]
//         this.stream.getAudioTracks().forEach(track=>track.stop());
//         this.stream=null

//     }
//             startRecording(){
//                if(this.isRecording)return
//                this.isRecording=true
//                this.startRef.innerHTML="Recording....";
//                this.playerRef.src='';
//                navigator.mediaDevices.getUserMedia(this.constraints)
//                .then(this.handleSucess.bind(this))
//                .catch(this.handleSucess.bind(this))
//             }

//             stopRecording(){
//                 if(!this.isRecording)return
//                 this.isRecording=false
//                 this.startRef.innerHTML="Record";
//                 this.recorderRef.pause();
//                 this.mediaRecorder.stop()
//             }

// }
// window.VoiceRecorder=new VoiceRecorder();
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);

function startRecording() {
	console.log("recordButton clicked");

	/*
		Simple constraints object, for more advanced audio features see
		https://addpipe.com/blog/audio-constraints-getusermedia/
	*/
    
    var constraints = { audio: true, video:false }

 	/*
    	Disable the record button until we get a success or fail from getUserMedia() 
	*/

	recordButton.disabled = true;
	stopButton.disabled = false;
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

		/*
			create an audio context after getUserMedia is called
			sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
			the sampleRate defaults to the one set in your OS for your playback device
		*/
		audioContext = new AudioContext();

		//update the format 
		document.getElementById("recorder").innerHTML="recorder: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz"

		/*  assign to gumStream for later use  */
		gumStream = stream;
		
		/* use the stream */
		input = audioContext.createMediaStreamSource(stream);

		/* 
			Create the Recorder object and configure to record mono sound (1 channel)
			Recording 2 channels  will double the file size
		*/
		rec = new Recorder(input,{numChannels:2})

		//start the recording process
		rec.record()

		console.log("Recording started");

	}).catch(function(err) {
	  	//enable the record button if getUserMedia() fails
    	recordButton.disabled = false;
    	stopButton.disabled = true;
    	
	});
}

function stopRecording() {
	console.log("stopButton clicked");

	//disable the stop button, enable the record too allow for new recordings
	stopButton.disabled = true;
	recordButton.disabled = false;
	

	//reset button just in case the recording is stopped while paused
	
	
	//tell the recorder to stop the recording
	rec.stop();

	//stop microphone access
	gumStream.getAudioTracks()[0].stop();

	//create the wav blob and pass it on to createDownloadLink
	rec.exportWAV(createDownloadLink);
}

async function createDownloadLink(blob) 
{
    // const file=new File(blob,'voice.wav',{type:blob.ty#})
    let formData = new FormData();
	let file=new File([blob],"my-rec.wav")

	formData.append('data', file);
	fetch('http://127.0.0.1:5000/', {
          method: 'POST',
          body: formData

      }).then(response => response.json()
      ).then(json => {
          console.log(json)
      });
	
    /* await $.ajax({
        method:"POST",
        // url: " gsjh",
        processData: false,
        contentType:false,
        async:false,
        data: formData
    }) */

    
}