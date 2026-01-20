//
//  Face.swift
//  FRS
//
//  Created by Lee, John on 7/14/19.
//  Copyright © 2019 Lee, John. All rights reserved.
//
import Foundation
import UIKit
import SafariServices

class Face {
    var boundingBox: [String:CGFloat]! //= ["height": 0.0,"left": 0.0,"top": 1.0,"width": 0.0]
    var name: String!
    var simularity: Float!
    var image: UIImage!
    var scene: UIImageView!
    
    init(name: String, simularity: Float, image: UIImage, scene: UIImageView) {
        self.name = name
        self.simularity = simularity
        self.image = image
        self.scene = scene
    }
}
