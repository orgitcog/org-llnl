//
//  CelebrityTableCell.swift
//  FRS
//
//  Created by Lee, John on 8/18/19.
//  Copyright © 2019 Lee, John. All rights reserved.
//

import UIKit

class CelebrityTableCell: UITableViewCell {
    @IBOutlet weak var croppedImage: UIImageView!
    @IBOutlet weak var lblName: UILabel!
    @IBOutlet weak var lblSimularity: UILabel!
    
    func setCell(face: Face) {
        self.croppedImage.image = face.image
        self.croppedImage.layer.cornerRadius = 10.0
        self.lblName.text = face.name
        self.lblSimularity.text = face.simularity > 0 ? face.simularity.description : ""
    }
}
