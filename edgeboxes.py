import cv2
import numpy as np
import argparse
import xml.etree.ElementTree as ET

def get_intersection_area(box1, box2):
    
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2], box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])

    if (x2 - x1 < 0) or (y2 - y1 < 0):
        return 0.0
    else:
        return (x2 - x1 + 1) * (y2 - y1 + 1)


def calculate_iou(proposal_boxes, gt_boxes):

    iou_qualified_boxes = []
    final_boxes = []
    for gt_box in gt_boxes:
        best_box_iou = 0
        best_box = 0
        area_gt_box = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        for prop_box in proposal_boxes:
            area_prop_box = (prop_box[2] - prop_box[0] + 1) * (prop_box[3] - prop_box[1] + 1)
            intersection_area = get_intersection_area(prop_box, gt_box)
            union_area = area_prop_box + area_gt_box - intersection_area
            iou = float(intersection_area) / float(union_area)
            if iou > 0.5:
                iou_qualified_boxes.append(prop_box)
                if iou > best_box_iou:
                    best_box_iou = iou
                    best_box = prop_box
        if best_box_iou != 0:
            final_boxes.append(best_box)
    return iou_qualified_boxes, final_boxes


def get_groundtruth_boxes(annoted_img_path):

    gt_boxes = []
    tree = ET.parse(annoted_img_path)
    root = tree.getroot()
    for items in root.findall('object/bndbox'):
        xmin = items.find('xmin')
        ymin = items.find('ymin')
        xmax = items.find('xmax')
        ymax = items.find('ymax')
        gt_boxes.append([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
    return gt_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image_path", default="./HW2_Data/JPEGImages/000480.jpg", type=str,
                        help="Enter the image path")
    parser.add_argument("annotated_image_path", default="./HW2_Data/Annotations/000480.xml", type=str,
                        help="Enter the annotated image path")
    args = parser.parse_args()

    img_path = args.input_image_path
    annotated_img_path = args.annotated_image_path
    img = cv2.imread(img_path)

    model_path = "model.yml.gz"
    edge_detection_obj = cv2.ximgproc.createStructuredEdgeDetection(model_path)

    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    edges = edge_detection_obj.detectEdges(np.float32(rgb_im)/255.0)

    orient_map = edge_detection_obj.computeOrientation(edges)

    edges = edge_detection_obj.edgesNms(edges, orient_map)
    cv2.imshow("Edges", edges)
    k = cv2.waitKey()
    cv2.destroyAllWindows()

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(100)
    edge_boxes.setAlpha(0.5)
    edge_boxes.setBeta(0.5)
    prop_boxes, scores = edge_boxes.getBoundingBoxes(edges, orient_map)

    boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in prop_boxes]

    output_img_proposal_top100 = img.copy()
    output_img_iou_qualified = img.copy()
    output_img_final = img.copy()

    gt_boxes = get_groundtruth_boxes(annotated_img_path)
    print("Number of Ground Truth Boxes = ", len(gt_boxes))

    for i in range(0, len(boxes)):
        top_x, top_y, bottom_x, bottom_y = boxes[i]
        cv2.rectangle(output_img_proposal_top100, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Output_Top_100_Proposals", output_img_proposal_top100)
    cv2.imwrite("./Results/Output_Top_100_Proposals.png", output_img_proposal_top100)
    cv2.waitKey()
    cv2.destroyAllWindows()

    iou_qualified_boxes, final_boxes = calculate_iou(boxes, gt_boxes)
    print("Number of Qualified Boxes with IOU > 0.5 = ", len(iou_qualified_boxes))
    print("Qualified Boxes = ", iou_qualified_boxes)

    for i in range(0, len(iou_qualified_boxes)):
        top_x, top_y, bottom_x, bottom_y = iou_qualified_boxes[i]
        cv2.rectangle(output_img_iou_qualified, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA)
    for i in range(0, len(gt_boxes)):
        top_x, top_y, bottom_x, bottom_y = gt_boxes[i]
        cv2.rectangle(output_img_iou_qualified, (top_x, top_y), (bottom_x, bottom_y), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Output_IOU_Qualified_Proposals", output_img_iou_qualified)
    cv2.imwrite("./Results/Output_IOU_Qualified_Proposals.png", output_img_iou_qualified)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print("Number of final boxes = ", len(final_boxes))
    print("Final boxes = ", final_boxes)

    recall = len(final_boxes) / len(gt_boxes)
    print("Recall = ", recall)

    for i in range(0, len(final_boxes)):
        top_x, top_y, bottom_x, bottom_y = final_boxes[i]
        cv2.rectangle(output_img_final, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA)
    for i in range(0, len(gt_boxes)):
        top_x, top_y, bottom_x, bottom_y = gt_boxes[i]
        cv2.rectangle(output_img_final, (top_x, top_y), (bottom_x, bottom_y), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Output_Final_Boxes", output_img_final)
    cv2.imwrite("./Results/output_img_final.png", output_img_final)
    cv2.waitKey()
    cv2.destroyAllWindows()
