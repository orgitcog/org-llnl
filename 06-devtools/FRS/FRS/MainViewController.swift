//
//  MainViewController.swift
//  FRS
//
//  Created by Lee, John on 7/14/19.
//  Copyright © 2019 Lee, John. All rights reserved.
//

import UIKit
import Foundation
import AVFoundation

class MainViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, AVCapturePhotoCaptureDelegate {
    
    @IBOutlet weak var previewView: UIView!
    @IBOutlet weak var btnAdd: UIButton!
    @IBOutlet weak var btnCelebrity: UIButton!
    @IBOutlet weak var btnEmployee: UIButton!
    @IBOutlet weak var btnAttribute: UIButton!
    
    var captureSession: AVCaptureSession!
    var capturedImage: AVCapturePhotoOutput!
    var image: UIImage!
    var segue: String!
    var videoPreviewLayer: AVCaptureVideoPreviewLayer!
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        captureSession.stopRunning()
        setupSession()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        setupSession()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession.stopRunning()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        switch (segue.identifier) {
        case "segueEmployee":
            let vc = segue.destination as! EmployeeViewController
            vc.image = image
        case "segueAttribute":
            let vc = segue.destination as! AttributeViewController
            vc.image = image
        case "segueCelebrity":
            let vc = segue.destination as! CelebrityViewController
            vc.image = image
        case "segueAdd":
            let vc = segue.destination as! AddViewController
            vc.image = image
        default:
            print("ERROR: Invalid segue selection")
        }
    }
    
    @IBAction func onAddClicked(_ sender: Any) {
        segue = "segueAdd"
        capturePhoto()
    }
    
    @IBAction func onAttributeClicked(_ sender: Any) {
        segue = "segueAttribute"
        capturePhoto()
    }
    
    @IBAction func onCelebrityClicked(_ sender: Any) {
        segue = "segueCelebrity"
        capturePhoto()
    }
    
    @IBAction func onEmployeeClicked(_ sender: Any) {
        segue = "segueEmployee"
        capturePhoto()
    }
    
    func capturePhoto() {
        let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        if (capturedImage != nil) {
            capturedImage.capturePhoto(with: settings, delegate: self)
        } else { // Perform segue with no captured image
            performSegue(withIdentifier: segue, sender: self)
        }
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard let imageData = photo.fileDataRepresentation()
            else { return }
        
        var orientation = UIImageOrientation.up
        if UIDevice.current.orientation == UIDeviceOrientation.landscapeLeft || (UIDevice.current.orientation == UIDeviceOrientation.faceUp && UIDevice.current.orientation.isLandscape){
            orientation = UIImageOrientation.up
        } else if UIDevice.current.orientation == UIDeviceOrientation.landscapeRight {
            orientation = UIImageOrientation.down
        } else if UIDevice.current.orientation == UIDeviceOrientation.portrait || (UIDevice.current.orientation == UIDeviceOrientation.faceUp && UIDevice.current.orientation.isPortrait){
            orientation = UIImageOrientation.right
        } else if UIDevice.current.orientation == UIDeviceOrientation.portraitUpsideDown{
            orientation = UIImageOrientation.left
        }
        let uiimage = UIImage(data: imageData)
        let cgimage = CIImage(image: uiimage!)
        image = UIImage(cgImage: (cgimage?.cgImage)!, scale: 1.0, orientation: orientation)
        performSegue(withIdentifier: segue, sender: self)
    }
    
    //Setup camera session
    func setupSession(){
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .medium
        guard let backCamera = AVCaptureDevice.default(for: AVMediaType.video)
            else {
                print("Unable to access back camera!")
                return
        }
        
        do {
            if(backCamera.isFocusModeSupported(.continuousAutoFocus)){
                try! backCamera.lockForConfiguration()
                backCamera.focusMode = .continuousAutoFocus
                backCamera.unlockForConfiguration()
            }
            let input = try AVCaptureDeviceInput(device: backCamera)

            capturedImage = AVCapturePhotoOutput()
            
            if captureSession.canAddInput(input) && captureSession.canAddOutput(capturedImage) {
                captureSession.addInput(input)
                captureSession.addOutput(capturedImage)
                setupLivePreview()
            }
        }
        catch let error  {
            print("Error Unable to initialize back camera:  \(error.localizedDescription)")
        }
    }
    
    //Setup live preview, handle rotations
    func setupLivePreview() {
        previewView.layer.sublayers?.removeAll()
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        
        videoPreviewLayer.videoGravity = .resizeAspect
        if UIDevice.current.orientation == UIDeviceOrientation.landscapeLeft {
            videoPreviewLayer.connection?.videoOrientation = .landscapeRight
        } else if UIDevice.current.orientation == UIDeviceOrientation.landscapeRight {
            videoPreviewLayer.connection?.videoOrientation = .landscapeLeft
        } else if UIDevice.current.orientation == UIDeviceOrientation.portrait {
            videoPreviewLayer.connection?.videoOrientation = .portrait
        } else if UIDevice.current.orientation == UIDeviceOrientation.portraitUpsideDown{
            videoPreviewLayer.connection?.videoOrientation = .portraitUpsideDown
        }
        previewView.layer.addSublayer(videoPreviewLayer)
        
        DispatchQueue.global(qos: .userInitiated).async { //[weak self] in
            self.captureSession.startRunning()

            DispatchQueue.main.async {
                self.videoPreviewLayer.frame = self.previewView.bounds
            }
        }
    }
}
