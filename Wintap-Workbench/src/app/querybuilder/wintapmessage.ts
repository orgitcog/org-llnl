import * as CodeMirror from 'codemirror';
import 'codemirror/addon/mode/overlay';

// Create an overlay mode for the SQL mode
const wintapmessage: CodeMirror.Mode<any> = {
    token: (stream: CodeMirror.StringStream, state: any): string | null => {
    const prevChar = stream.string.charAt(stream.start - 1);
    const validPrevChar = prevChar === "." || prevChar === " ";

    // syntax highlighting rules
    // top-level
      if (stream.match(/^WintapMessage/, true)) {
        return "atom";
      }
      if (stream.match(/^PID\b/, true)) {
        return "number";
      }
      if (stream.match(/^MessageType\b/, true)) {
        return "string";
      }
      if (stream.match(/^ActivityType\b/, true)) {
        return "string";
      }
      if (stream.match(/^EventTime\b/, true)) {
        return "number";
      }
      // Process
      if (stream.match(/^Process\b/, true) && validPrevChar) {
        return "atom";
      }  
      if (stream.match(/^ParentPID\b/, true)) {
        return "number";
      }
      if (stream.match(/^Name\b/, true)) {
        return "string";
      }
      if (stream.match(/^Path\b/, true)) {
        return "string";
      }
      if (stream.match(/^CommandLine\b/, true)) {
        return "string";
      }
      if (stream.match(/^Arguments\b/, true)) {
        return "string";
      }
      if (stream.match(/^User\b/, true)) {
        return "string";
      }
      if (stream.match(/^ExitCode\b/, true)) {
        return "number";
      }
      // TCP
      if (stream.match(/^TcpConnection\b/, true) && validPrevChar) {
        return "atom";
      }  
      if (stream.match(/^SourceAddress\b/, true)) {
        return "string";
      }
      if (stream.match(/^SourcePort\b/, true)) {
        return "number";
      }
      if (stream.match(/^DestinationAddress\b/, true)) {
        return "string";
      }
      if (stream.match(/^DestinationPort\b/, true)) {
        return "number";
      }
      // UDP
      if (stream.match(/^UdpPacket\b/, true) && validPrevChar) {
        return "atom";
      }  
      if (stream.match(/^SourceAddress\b/, true)) {
        return "string";
      }
      if (stream.match(/^SourcePort\b/, true)) {
        return "number";
      }
      if (stream.match(/^DestinationAddress\b/, true)) {
        return "string";
      }
      if (stream.match(/^DestinationPort\b/, true)) {
        return "number";
      }
      if (stream.match(/^PacketSize\b/, true)) {
        return "number";
      }
      if (stream.match(/^FailureCode\b/, true)) {
        return "string";
      }
       // ImageLoad
       if (stream.match(/^ImageLoad\b/, true) && validPrevChar) {
        return "atom";
      }  
      if (stream.match(/^FileName\b/, true)) {
        return "string";
      }
      if (stream.match(/^BuildTime\b/, true)) {
        return "number";
      }
      if (stream.match(/^ImageChecksum\b/, true)) {
        return "number";
      }
      if (stream.match(/^ImageSize\b/, true)) {
        return "number";
      }
      if (stream.match(/^DefaultBase\b/, true)) {
        return "string";
      }
      if (stream.match(/^ImageBase\b/, true)) {
        return "string";
      }
      if (stream.match(/^MD5\b/, true)) {
        return "string";
      }
      // File
      if (stream.match(/^FileActivity\b/, true) && validPrevChar) {
        return "atom";
      }  
      if (stream.match(/^BytesRequested\b/, true)) {
        return "number";
      }
      // Registry
      if (stream.match(/^RegActivity\b/, true) && validPrevChar) {
        return "atom";
      }  
      if (stream.match(/^DataType\b/, true)) {
        return "string";
      }
      if (stream.match(/^ValueName\b/, true)) {
        return "string";
      }
      if (stream.match(/^Data\b/, true)) {
        return "string";
      }


      stream.next();
      return null;
    },
  };  

// Register the overlay mode with CodeMirror..
CodeMirror.defineMode('wintapmessage', () => {
    console.log('registered wintapmessage with codemirror')
    return wintapmessage;
  });

export { wintapmessage };
