//
//  EmployeeViewController.swift
//  FRS
//
//  Created by Lee, John on 7/14/19.
//  Copyright © 2019 Lee, John. All rights reserved.
//
import UIKit
import Foundation
import SafariServices
import AWSRekognition
import AWSDynamoDB

class EmployeeViewController: UIViewController, UINavigationControllerDelegate, SFSafariViewControllerDelegate,  UITableViewDataSource, UITableViewDelegate {
    
    @IBOutlet weak var tableView: UITableView!
    @IBOutlet weak var capturedImage: UIImageView!
    
    let activityIndicator = UIActivityIndicatorView(activityIndicatorStyle: UIActivityIndicatorView.Style.whiteLarge)

    var dynamoDB: AWSDynamoDB?
    var faces: [Face] = []
    var image: UIImage!
    var imageData: Data!
    var rekognitionObject: AWSRekognition?
    var rekogCollectionId = "faces"     // Rekogntion Collection Id
    var rekogThreshold = 60             // Threshold for simularity match 0 - 100
    var rekogMatches = 10               // Total matches to return by Rekognition
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        if image == nil {
            print ("ERROR! No image was received. Loading default image!")
            image = #imageLiteral(resourceName: "bezos")
        }
        capturedImage.image = image
        image = image.makePortrait()
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
        return faces.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "EmployeeTableCell") as! EmployeeTableCell
        let face = faces[indexPath.row]
        cell.setCell(face: face)
        if(face.simularity < 60.0) {
            cell.lblSimularity.textColor = UIColor.red
        }
        else if(face.simularity >= 85.0) {
            cell.lblSimularity.textColor = UIColor.green
        }
        else {
            cell.lblSimularity.textColor = UIColor.orange
        }
        return cell
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
            if (result!.faceDetails!.count > 0) { // Faces found! Process them
                print("Number of faces detected in image: \(result!.faceDetails!.count)")
                
                for (_, face) in result!.faceDetails!.enumerated(){
                    if(face.confidence!.intValue > 50) { // if its really face
                        let viewHeight = face.boundingBox?.height  as! CGFloat
                        let viewWidth = face.boundingBox?.width as! CGFloat
                        let toRect = CGRect(x: face.boundingBox?.left as! CGFloat, y: face.boundingBox?.top as! CGFloat, width: viewWidth, height:viewHeight)
                        let croppedImage = self.cropImage(self.image!, toRect: toRect, viewWidth: viewWidth, viewHeight: viewHeight)
                        let croppedFace: Data = UIImageJPEGRepresentation(croppedImage!, 1.0)!
                        
                        self.rekognizeFace(faceImageData: croppedFace, detectedface: face, croppedFace: croppedImage!)
                    }
                }
            }
            else {
                print("No faces were detected in this image.")
                let workItem = DispatchWorkItem {
                    [weak self] in
                    let face = Face(name: "No faces detected", simularity: 0.0, image: #imageLiteral(resourceName: "error"), scene: self!.capturedImage)
                    self?.faces.append(face)
                    self?.reloadList()
                    self?.activityIndicator.stopAnimating()
                }
                DispatchQueue.main.async(execute: workItem)
            }
        }
    }
    
    func rekognizeFace(faceImageData: Data, detectedface: AWSRekognitionFaceDetail, croppedFace: UIImage) {
        rekognitionObject = AWSRekognition.default()
        let faceImageAWS = AWSRekognitionImage()
        faceImageAWS?.bytes = faceImageData
        
        let imagerequest = AWSRekognitionSearchFacesByImageRequest()
        imagerequest?.collectionId = rekogCollectionId
        imagerequest?.faceMatchThreshold = rekogThreshold as NSNumber
        imagerequest?.maxFaces = rekogMatches as NSNumber
        imagerequest?.image = faceImageAWS
        
        let faceInImage = Face(name: "Unknown", simularity: 0.0, image: croppedFace, scene:  self.capturedImage)
        
        faceInImage.boundingBox = ["height":detectedface.boundingBox?.height, "left":detectedface.boundingBox?.left, "top":detectedface.boundingBox?.top, "width":detectedface.boundingBox?.width] as? [String : CGFloat]
        
        rekognitionObject?.searchFaces(byImage: imagerequest!) {
            (result, error) in
            if error != nil {
                print(error!)
                return
            }
            if (result != nil && result!.faceMatches!.count > 0) {
                print("Total faces matched by Rekogition: \(result!.faceMatches!.count)")
                print ("Attempting to retrieve face information from DynamoDB")
                
                for (_, face) in result!.faceMatches!.enumerated() {
                    faceInImage.simularity = face.similarity!.floatValue
                    
                    // Get face full name from DynamoDB
                    self.dynamoDB = AWSDynamoDB.default()
                    let iteminput = AWSDynamoDBQueryInput()
                    iteminput?.indexName = "faceid-index"
                    iteminput?.tableName = "index-face"
                    iteminput?.keyConditionExpression = "faceid = :v1"
                    let value = AWSDynamoDBAttributeValue()
                    value?.s = face.face?.faceId
                    iteminput?.expressionAttributeValues = [":v1" : value!]
                    
                    self.dynamoDB?.query(iteminput!) {
                        (result, err) in
                        
                        if let error = err as NSError? {
                            print("Unable to get face name from dynamo: \(error)")
                            faceInImage.name = "Name Missing"
                        }
                        else {
                            for (_, value1) in
                                result!.items!.enumerated() {
                                    for (_, value2) in value1.enumerated() {
                                        if (value2.key == "name"){
                                            faceInImage.name = value2.value.s!
                                        }
                                    }
                            }
                        }
                        print ("\(face.face?.faceId ?? "") | \(faceInImage.name ?? "unavailable") | \(face.similarity!.floatValue)")
                        let workItem = DispatchWorkItem {
                            [weak self] in
                            let match = Face(name: faceInImage.name!, simularity: face.similarity!.floatValue, image: croppedFace, scene:  self!.capturedImage)
                            self?.faces.append(match)
                            self?.reloadList()
                            self?.activityIndicator.stopAnimating()
                        }
                        DispatchQueue.main.async(execute: workItem)
                    }
                }
            }
            else {
                print("Rekognition could not match any faces")
                let workItem = DispatchWorkItem {
                    [weak self] in
                    self?.faces.append(faceInImage)
                    self?.reloadList()
                    self?.activityIndicator.stopAnimating()
                }
                DispatchQueue.main.async(execute: workItem)
            }
        }
    }
    
    func cropImage(_ inputImage: UIImage, toRect cropRect: CGRect, viewWidth: CGFloat, viewHeight: CGFloat) -> UIImage? {
        let cropZone = CGRect(x:cropRect.origin.x * inputImage.size.width,
                              y:cropRect.origin.y * inputImage.size.height,
                              width:cropRect.size.width * inputImage.size.width,
                              height:cropRect.size.height * inputImage.size.height)
        
        guard let cutImageRef: CGImage = inputImage.cgImage?.cropping(to:cropZone)
            else {
                return nil
        }
        
        return UIImage(cgImage: cutImageRef)
    }

    func reloadList() {
        faces.sort() { $0.simularity > $1.simularity }
        tableView.reloadData();
    }
}
