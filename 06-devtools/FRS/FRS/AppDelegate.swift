//
//  AppDelegate.swift
//  FRS
//
//  Created by Lee, John on 7/14/19.
//  Copyright © 2019 Lee, John. All rights reserved.
//

import UIKit
import AWSCore

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
    
    var window: UIWindow?
    
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        
        // Initialize Identity Provider
        let credentialsProvider = AWSCognitoCredentialsProvider(
            regionType: .USWest2,
            identityPoolId: "poolId")
        let configuration = AWSServiceConfiguration(
            region: .USWest2,
            credentialsProvider: credentialsProvider)
        AWSServiceManager.default().defaultServiceConfiguration = configuration
        
        return true
    }
    
    func applicationWillResignActive(_ application: UIApplication) {}
    
    func applicationDidEnterBackground(_ application: UIApplication) {}
    
    func applicationWillEnterForeground(_ application: UIApplication) {}
    
    func applicationDidBecomeActive(_ application: UIApplication) {}
    
    func applicationWillTerminate(_ application: UIApplication) {}
}
