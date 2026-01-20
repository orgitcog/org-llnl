//
//  AttributeViewController.swift
//  FRS
//
//  Created by Lee, John on 8/17/19.
//  Copyright © 2019 Lee, John. All rights reserved.
//

import UIKit
import Foundation
import SafariServices
import AWSRekognition

class AttributeViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {

    @IBOutlet weak var tableView: UITableView!
    @IBOutlet weak var capturedImage: UIImageView!
    
    let activityIndicator = UIActivityIndicatorView(activityIndicatorStyle: UIActivityIndicatorView.Style.whiteLarge)
    
    var attributes: [Attribute] = []
    var image: UIImage!
    var imageData: Data!
    var orientation = 0
    var rekognitionObject: AWSRekognition?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        if image == nil {
            print ("ERROR! No image was received. Loading default image!")
            image = #imageLiteral(resourceName: "melindagates")
        }
        capturedImage.image = image
        imageData = UIImagePNGRepresentation(image)!
        
        tableView.delegate = self
        tableView.dataSource = self
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        
        activityIndicator.color = .darkGray
        activityIndicator.center = CGPoint(x: tableView.bounds.size.width/2, y: tableView.bounds.size.height/3)
        tableView.addSubview(activityIndicator)
        activityIndicator.startAnimating()
        
        detectFaces()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return attributes.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "AttributeTableCell") as! AttributeTableCell
        let attribute = attributes[indexPath.row]
        cell.setCell(attribute: attribute)
        return cell
    }
    
    func detectFaces() {
        rekognitionObject = AWSRekognition.default()
        let faceImageAWS = AWSRekognitionImage()
        faceImageAWS?.bytes = imageData
        
        let detectfacesrequest = AWSRekognitionDetectFacesRequest()
        detectfacesrequest?.image = faceImageAWS
        detectfacesrequest?.attributes = ["ALL"]
        
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
                    let match = Attribute(name: "More than one face found!", value: "", confidence: 0.0)
                    self?.attributes.append(match)
                    self?.reloadList()
                    self?.activityIndicator.stopAnimating()
                }
                DispatchQueue.main.async(execute: workItem)
            }
            else if (result!.faceDetails!.count > 0) {
                print("Face detected. Grabbing attributes")
                for (_, face) in result!.faceDetails!.enumerated() {
                    // Photo Quality - debugging only
                    if (face.quality != nil) {
                        print("Brightness: \(face.quality!.brightness?.description ?? "Unknow") | Sharpness: \(face.quality!.sharpness?.description ?? "Unknown")")
                    }
                    // Age
                    if (face.ageRange != nil && face.ageRange!.low != nil && face.ageRange!.high != nil) {
                        let match = Attribute(name: "Age", value: "\(face.ageRange!.low!.decimalValue) to \(face.ageRange!.high!.decimalValue)", confidence: 99.0)
                        self.attributes.append(match)
                    }
                    // Gender
                    if (face.gender != nil) {
                        let value = face.gender!.value.rawValue > 1 ? "Female" : "Male"
                        let match = Attribute(name: "Gender", value: value, confidence: face.gender!.confidence as! Float)
                        self.attributes.append(match)
                    }
                    // Beard
                    if (face.beard != nil && face.beard!.value != nil) {
                        let value = face.beard!.value!.boolValue ? "Yes" : "No"
                        let match = Attribute(name: "Beard", value: value, confidence: face.beard!.confidence as! Float)
                        self.attributes.append(match)
                    }
                    // Glasses
                    if (face.eyeglasses != nil && face.eyeglasses!.value != nil) {
                        let value = face.eyeglasses!.value!.boolValue ? "Yes" : "No"
                        let match = Attribute(name: "Eye Glasses", value: "\(value)", confidence: face.eyeglasses?.confidence as! Float)
                        self.attributes.append(match)
                    }
                    // Eyes Open
                    if (face.eyesOpen != nil && face.eyesOpen!.value != nil) {
                        let value = face.eyesOpen!.value!.boolValue ? "Yes" : "No"
                        let match = Attribute(name: "Eyes Open", value: "\(value)", confidence: face.eyesOpen?.confidence as! Float)
                        self.attributes.append(match)
                    }
                    // Mouth Open
                    if (face.mouthOpen != nil && face.mouthOpen!.value != nil) {
                        let value = face.mouthOpen!.value!.boolValue ? "Yes" : "No"
                        let match = Attribute(name: "Mouth Open", value: "\(value)", confidence: face.mouthOpen?.confidence as! Float)
                        self.attributes.append(match)
                    }
                    // Mustache
                    if (face.mustache != nil && face.mustache!.value != nil) {
                        let value = face.mustache!.value!.boolValue ? "Yes" : "No"
                        let match = Attribute(name: "Mustache", value: "\(value)", confidence: face.mustache?.confidence as! Float)
                        self.attributes.append(match)
                    }
                    // Smile
                    if (face.smile != nil && face.smile!.value != nil) {
                        let value = face.smile!.value!.boolValue ? "Yes" : "No"
                        let match = Attribute(name: "Smile", value: "\(value)", confidence: face.smile?.confidence as! Float)
                        self.attributes.append(match)
                    }
                    // Sunglasses
                    if (face.sunglasses != nil && face.sunglasses!.value != nil) {
                        let value = face.sunglasses!.value!.boolValue ? "Yes" : "No"
                        let match = Attribute(name: "Sunglasses", value: "\(value)", confidence: face.sunglasses?.confidence as! Float)
                        self.attributes.append(match)
                    }
                    // Emotions
                    if (face.emotions != nil && face.emotions!.count > 0) {
                        print ("Found emotions")
                        for (_, emotion) in face.emotions!.enumerated() {
                            let value = emotion.types.rawValue
                            var emotionName = ""
                            switch (value) {
                            case 0:
                                emotionName = "Unknown"
                            case 1:
                                emotionName = "Happy"
                            case 2:
                                emotionName = "Sad"
                            case 3:
                                emotionName = "Angry"
                            case 4:
                                emotionName = "Confused"
                            case 5:
                                emotionName = "Disgusted"
                            case 6:
                                emotionName = "Surprised"
                            default:
                                emotionName = "Calm"
                                break
                            }
                            if (emotion.confidence!.int32Value > 10) {
                                let match = Attribute(name: "Emotion", value: "\(emotionName)", confidence: emotion.confidence as! Float)
                                self.attributes.append(match)
                            }
                        }
                    } else {
                        print ("No emotions")
                    }
                    // Print Results
                    let workItem = DispatchWorkItem {
                        [weak self] in
                        self?.reloadList()
                        self?.activityIndicator.stopAnimating()
                    }
                    DispatchQueue.main.async(execute: workItem)
                }
            }
            else {
                print("No faces were detected in this image.")
                let workItem = DispatchWorkItem {
                    [weak self] in
                    let attribute = Attribute(name: "No faces detected", value: "", confidence: 0.0)
                    self?.attributes.append(attribute)
                    self?.reloadList()
                    self?.activityIndicator.stopAnimating()
                }
                DispatchQueue.main.async(execute: workItem)
            }
        }
    }
    
    func reloadList() {
        attributes.sort() { $0.name > $1.name }
        tableView.reloadData();
    }
}
