//
//  AddViewController.swift
//  FRS
//
//  Created by Lee, John on 8/18/19.
//  Copyright © 2019 Lee, John. All rights reserved.
//

import UIKit
import Foundation
import AWSRekognition
import AWSS3

class AddViewController: UIViewController, UITextFieldDelegate {

    @IBOutlet weak var capturedImage: UIImageView!
    @IBOutlet weak var btnSave: UIButton!
    @IBOutlet weak var tbName: UITextField!
    @IBOutlet weak var lblDescription: UILabel!
    
    let S3BucketName = "llnl-sapo-frs"
    var image: UIImage!
    var imageData: Data!
    var rekognitionObject: AWSRekognition?
    var completionHandler: AWSS3TransferUtilityUploadCompletionHandlerBlock?
    
    @IBAction func onClickSave(_ sender: Any) {
        tbName.isUserInteractionEnabled = false
        btnSave.isEnabled = false
        self.uploadPhoto()
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        tbName.delegate = self

        if image == nil {
            print ("ERROR! No image was received. Loading default image!")
            image = #imageLiteral(resourceName: "susanwojcicki")
        }
        capturedImage.image = image
        image = image.makePortrait()
        imageData = UIImagePNGRepresentation(image)!
        
        self.completionHandler = { (task, error) -> Void in
            DispatchQueue.main.async(execute: {
                if let error = error {
                    self.showMessage(message: "Failed to upload image!")
                    print("ERROR: \(error)")
                }
                else{
                    self.showMessage(message: "Successfully uploaded image!")
                }
            })
        }
        
        detectFaces()
    }
    
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        tbName.resignFirstResponder()
        return true
    }
    
    func detectFaces() {
        rekognitionObject = AWSRekognition.default()
        let faceImageAWS = AWSRekognitionImage()
        faceImageAWS?.bytes = imageData
        
        let detectfacesrequest = AWSRekognitionDetectFacesRequest()
        detectfacesrequest?.image = faceImageAWS
        
        rekognitionObject?.detectFaces(detectfacesrequest!) {
            (result, error) in
            if error != nil {
                print(error!)
                return
            }
            if (result!.faceDetails!.count > 1) { // Faces found! Process them
                print("More than one face found: \(result!.faceDetails!.count)")
                let workItem = DispatchWorkItem {
                    [weak self] in
                    self?.showMessage(message: "More than one face detected!")
                }
                DispatchQueue.main.async(execute: workItem)
            }
            else if (result!.faceDetails!.count > 0) {
                print("Face detected. Grabbing attributes")
                for (_, face) in result!.faceDetails!.enumerated() {
                    // Photo Quality needs to be good
                    if (face.quality != nil && Int(truncating: face.quality!.brightness!) > 70 && Int(truncating: face.quality!.sharpness!) > 70) {
                        print("Brightness: \(face.quality!.brightness?.description ?? "Unknow") | Sharpness: \(face.quality!.sharpness?.description ?? "Unknown")")
                    } else {
                        let workItem = DispatchWorkItem {
                            [weak self] in
                            self?.showMessage(message: "Photo Quality is Poor!")
                        }
                        DispatchQueue.main.async(execute: workItem)
                    }
                }
            }
            else {
                print("No faces were detected in this image.")
                let workItem = DispatchWorkItem {
                    [weak self] in
                    self?.showMessage(message: "No faces detected!")
                }
                DispatchQueue.main.async(execute: workItem)
            }
        }
    }
    
    func showMessage(message: String) {
        lblDescription.text = message
    }
    
    func uploadPhoto() {
        if (tbName.text == nil || tbName.text!.count < 3) {
            DispatchQueue.main.async(execute: {
                self.showMessage(message: "Please enter full name")
            })
            return
        }
        
        let filename = tbName.text!.replacingOccurrences(of: " ", with: "_") + ".png"
        let expression = AWSS3TransferUtilityUploadExpression()
        let transferUtility = AWSS3TransferUtility.default()
        
        transferUtility.uploadData(
            imageData,
            bucket: S3BucketName,
            key: filename,
            contentType: "image/png",
            expression: expression,
            completionHandler: completionHandler
        ).continueWith { (task) -> AnyObject? in
            if let error = task.error {
                self.showMessage(message: "Error uploading image!")
                print("ERROR UPLOAD: \(error.localizedDescription)")
            }
            if let _ = task.result {
                self.showMessage(message: "Uploading image...")
            }
            return nil;
        }
    }
}
