//
//  Attribute.swift
//  FRS
//
//  Created by Lee, John on 8/22/19.
//  Copyright © 2019 Lee, John. All rights reserved.
//

import Foundation

class Attribute {
    var name: String!
    var value: String!
    var confidence: Float!
    
    init(name: String, value: String, confidence: Float) {
        self.name = name
        self.value = value
        self.confidence = confidence
    }
}
