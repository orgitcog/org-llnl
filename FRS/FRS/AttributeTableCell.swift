//
//  DetailsTableCell.swift
//  FRS
//
//  Created by Lee, John on 8/17/19.
//  Copyright © 2019 Lee, John. All rights reserved.
//

import UIKit

class AttributeTableCell: UITableViewCell {
    
    @IBOutlet weak var lblName: UILabel!
    @IBOutlet weak var lblValue: UILabel!
    @IBOutlet weak var lblConfidence: UILabel!
    
    func setCell(attribute: Attribute) {
        self.lblName.text = attribute.name
        self.lblValue.text = attribute.value
        self.lblConfidence.text = "\(Int(floor(attribute.confidence)))%"
    }
}
