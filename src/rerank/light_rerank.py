import cv2, numpy as np

def count_inliers_orb(img1_path,img2_path,max_features=1000):
    img1=cv2.imread(img1_path,cv2.IMREAD_GRAYSCALE); img2=cv2.imread(img2_path,cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None: return 0
    orb=cv2.ORB_create(nfeatures=max_features); kp1,des1=orb.detectAndCompute(img1,None); kp2,des2=orb.detectAndCompute(img2,None)
    if des1 is None or des2 is None: return 0
    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True); matches=bf.match(des1,des2)
    if len(matches)<4: return 0
    matches=sorted(matches,key=lambda x:x.distance)
    pts1=np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2=np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    try:
        H,mask=cv2.findHomography(pts1,pts2,cv2.RANSAC,5.0)
        if mask is None: return 0
        return int(mask.sum())
    except Exception:
        return 0

def rerank_topk_by_inliers(query_path,db_paths_topk):
    scores=[count_inliers_orb(query_path,p) for p in db_paths_topk]; order=sorted(range(len(scores)), key=lambda i:-scores[i]); return order,scores
