//
//  CelebrityViewController.swift
//  FRS
//
//  Created by Lee, John on 8/18/19.
//  Copyright © 2019 Lee, John. All rights reserved.
//

import UIKit
import Foundation
import SafariServices
import AWSRekognition

class CelebrityViewController: UIViewController, UINavigationControllerDelegate, SFSafariViewControllerDelegate,  UITableViewDataSource, UITableViewDelegate {

    @IBOutlet weak var capturedImage: UIImageView!
    @IBOutlet weak var tableView: UITableView!
    
    let activityIndicator = UIActivityIndicatorView(activityIndicatorStyle: UIActivityIndicatorView.Style.whiteLarge)
    
    var faces: [Face] = []
    var image: UIImage!
    var imageData: Data!
    var rekognitionObject: AWSRekognition?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        if image == nil {
            print ("ERROR! No image was received. Loading default image!")
            image = #imageLiteral(resourceName: "bradpitt")
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
        let cell = tableView.dequeueReusableCell(withIdentifier: "CelebrityTableCell") as! CelebrityTableCell
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
                self.rekognizeFace()
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
    
    func rekognizeFace() {
        rekognitionObject = AWSRekognition.default()
        let faceImageAWS = AWSRekognitionImage()
        faceImageAWS?.bytes = imageData
        
        let celebRequest = AWSRekognitionRecognizeCelebritiesRequest()
        celebRequest?.image = faceImageAWS
        
        rekognitionObject?.recognizeCelebrities(celebRequest!){
            (result, error) in
            if error != nil {
                print(error!)
                return
            }
            if (result != nil && result!.celebrityFaces!.count > 0) {
                print("Total celebrities matched by Rekogition: \(result!.celebrityFaces!.count)")
                
                for (_, celebFace) in result!.celebrityFaces!.enumerated(){
                    print ("\(celebFace.name ?? "") | \(celebFace.face!.confidence ?? 0)")
                    
                    let viewHeight = celebFace.face!.boundingBox?.height  as! CGFloat
                    let viewWidth = celebFace.face!.boundingBox?.width as! CGFloat
                    let toRect = CGRect(x: celebFace.face!.boundingBox?.left as! CGFloat, y: celebFace.face!.boundingBox?.top as! CGFloat, width: viewWidth, height:viewHeight)
                    let croppedImage = self.cropImage(self.image!, toRect: toRect, viewWidth: viewWidth, viewHeight: viewHeight)
                    let _: Data = UIImageJPEGRepresentation(croppedImage!, 1.0)!
                    
                    let workItem = DispatchWorkItem {
                        [weak self] in
                        let match = Face(name: celebFace.name!, simularity: celebFace.face!.confidence as! Float, image: croppedImage!, scene:  self!.capturedImage)
                        self?.faces.append(match)
                        self?.reloadList()
                        self?.activityIndicator.stopAnimating()
                    }
                    DispatchQueue.main.async(execute: workItem)
                }
            }
            else {
                print("Rekognition could not match any celebrity faces")
                let workItem = DispatchWorkItem {
                    [weak self] in
                    let faceInImage = Face(name: "Unknown", simularity: 0.0, image: (self?.capturedImage.image)!, scene:  self!.capturedImage)
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
