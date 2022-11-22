// window.onload=function(){
//     // document.querySelector('h2').style.color='red';

// }

class VoiceRecorder{
    constructor(){
        if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
            console.log("Get user media suported");

        }else{
            console.log("Get user media not suported");

        }
        this.mediaRecorder
        this.stream
        this.chunks =[]
        this.isRecording = false

        this.recorderRef=document.querySelector("#recorder");
        this.playerRef=document.querySelector("#player");
        this.startRef=document.querySelector("#start");
        this.stopRef=document.querySelector("#stop");

        this.startRef.onclick=this.startRecording.bind(this);
        this.stopRef.onclick=this.stopRecording.bind(this);

        this.constraints ={
            audio:true,
            video:false
        }

    }
    //handle sucess
    handleSucess(stream){
        this.stream=stream
        this.stream.onclick=()=>{
            console.log("stream ended");
        }
        this.recorderRef.srcObject=this.stream
        this.mediaRecorder=new MediaRecorder(this.stream)
        this.mediaRecorder.ondataavailable=this.onMediaRecorderDataAvailable.bind(this);
        this.mediaRecorder.onstop=this.onMediaRecorderStop.bind(this);
        this.recorderRef.play();
        this.mediaRecorder.start();
    }
    onMediaRecorderDataAvailable(e){this.chunks.push(e.data)}
    onMediaRecorderStop(e){
        const blob = new Blob(this.chunks,{'type':'audio/wav; codesc=opus'});
        const audioUrl= window.URL.createObjectURL(blob);
        this.playerRef.src=audioUrl;
        this.chunks=[]
        this.stream.getAudioTracks().forEach(track=>track.stop());
        this.stream=null

    }
            startRecording(){
               if(this.isRecording)return
               this.isRecording=true
               this.startRef.innerHTML="Recording....";
               this.playerRef.src='';
               navigator.mediaDevices.getUserMedia(this.constraints)
               .then(this.handleSucess.bind(this))
               .catch(this.handleSucess.bind(this))
            }

            stopRecording(){
                if(!this.isRecording)return
                this.isRecording=false
                this.startRef.innerHTML="Record";
                this.recorderRef.pause();
                this.mediaRecorder.stop()
            }

}
window.VoiceRecorder=new VoiceRecorder();
 

